/**
 * Reads single features' geometry from remote GeoParquet with HTTP range requests (hyparquet).
 *
 * Each file's footer is read once and kept. A read fetches, for each row group holding a requested row, only the
 * `id` and `geometry` column chunks, from the first to the last requested row in that group. Geometry stays WKB.
 */

import {
  type AsyncBuffer,
  asyncBufferFromUrl,
  type FileMetaData,
  parquetMetadataAsync,
  parquetReadObjects,
} from 'hyparquet';
import { compressors } from 'hyparquet-compressors';

interface Opened {
  file: AsyncBuffer;
  metadata: FileMetaData;
  starts: number[]; // first row of each row group, plus the row count
}

export interface Feature {
  id: string;
  wkb: Uint8Array;
}

export class GeoReader {
  /** Bytes received so far (GET responses). */
  bytes = 0;
  private files = new Map<string, Promise<Opened>>();

  private readonly fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const res = await fetch(input, init);
    if ((init?.method ?? 'GET') === 'GET') this.bytes += Number(res.headers.get('content-length') ?? 0);
    return res;
  };

  private open(url: string): Promise<Opened> {
    if (!this.files.has(url)) {
      const opened = (async () => {
        const file = await asyncBufferFromUrl({ url, fetch: this.fetch });
        const metadata = await parquetMetadataAsync(file, { geoparquet: false });
        const starts = [0];
        for (const rg of metadata.row_groups) starts.push(starts[starts.length - 1] + Number(rg.num_rows));
        return { file, metadata, starts };
      })();
      opened.catch(() => this.files.delete(url));
      this.files.set(url, opened);
    }
    return this.files.get(url)!;
  }

  private async groups(url: string, rows: number[]): Promise<Map<number, number[]>> {
    const { starts } = await this.open(url);
    const groups = new Map<number, number[]>();
    for (const row of rows) {
      let g = 0;
      while (starts[g + 1] <= row) g++;
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g)!.push(row);
    }
    return groups;
  }

  /** Compressed bytes a read of these rows will fetch (the id and geometry column chunks of their row groups). */
  async estimate(url: string, rows: number[]): Promise<number> {
    const { metadata } = await this.open(url);
    let total = 0;
    for (const g of (await this.groups(url, rows)).keys()) {
      for (const c of metadata.row_groups[g].columns) {
        const name = c.meta_data?.path_in_schema[0];
        if (name === 'id' || name === 'geometry') total += Number(c.meta_data?.total_compressed_size ?? 0);
      }
    }
    return total;
  }

  /** id and WKB geometry of the given rows, by row number. */
  async read(url: string, rows: number[]): Promise<Map<number, Feature>> {
    const { file, metadata } = await this.open(url);
    const out = new Map<number, Feature>();
    await Promise.all(
      [...(await this.groups(url, rows)).values()].map(async (rs) => {
        const rowStart = Math.min(...rs);
        const rowEnd = Math.max(...rs) + 1;
        const objs = await parquetReadObjects({
          file,
          metadata,
          compressors,
          rowStart,
          rowEnd,
          columns: ['id', 'geometry'],
          geoparquet: false,
          utf8: false, // keep WKB as bytes (id is a STRING column and still decodes as text)
        });
        for (const row of rs) {
          const o = objs[row - rowStart];
          const id = typeof o.id === 'string' ? o.id : new TextDecoder().decode(o.id);
          out.set(row, { id, wkb: o.geometry });
        }
      }),
    );
    return out;
  }
}
