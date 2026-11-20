/**
 * The plan model in the browser: SmolLM2-135M fine-tuned on gazet plans and place extraction, as a GGUF run by
 * wllama (llama.cpp's server core compiled to WebAssembly). Plans are decoded under the plan JSON schema with IDs
 * limited to the candidates, exactly as gazet does with llama-server.
 */

import { type LoadModelParams, Wllama } from '@wllama/wllama';
import wllamaWasm from '@wllama/wllama/esm/wasm/wllama.wasm?url';
import { type Candidate, parsePlaces, planUserPrompt } from './format.ts';
import { planSchema } from './plan.ts';
import { PLACES_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT } from './prompts.ts';

export interface Answer {
  text: string;
  promptTokens: number;
  outputTokens: number;
  seconds: number;
}

export class Llm {
  private constructor(private readonly wllama: Wllama) {}

  static async load(
    modelUrl: string,
    onProgress: (loaded: number, total: number) => void,
    params: LoadModelParams = {},
  ): Promise<Llm> {
    const wllama = new Wllama({ default: wllamaWasm }, { suppressNativeLog: true });
    await wllama.loadModelFromUrl(new URL(modelUrl, location.href).href, {
      // Two slots: one keeps the place-extraction prompt cached, the other serves plans.
      n_ctx: 8192,
      n_parallel: 2,
      progressCallback: ({ loaded, total }) => onProgress(loaded, total),
      ...params,
    });
    return new Llm(wllama);
  }

  async chat(system: string, user: string, schema?: object): Promise<Answer> {
    const t = performance.now();
    const res = await this.wllama.createChatCompletion({
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
      temperature: 0,
      max_tokens: 512,
      cache_prompt: true,
      ...(schema ? { response_format: { type: 'json_schema', json_schema: { name: 'plan', schema, strict: true } } } : {}),
    });
    return {
      text: res.choices[0].message.content ?? '',
      promptTokens: res.usage?.prompt_tokens ?? 0,
      outputTokens: res.usage?.completion_tokens ?? 0,
      seconds: (performance.now() - t) / 1000,
    };
  }

  async places(query: string): Promise<{ places: string[]; answer: Answer }> {
    const answer = await this.chat(PLACES_SYSTEM_PROMPT, query);
    return { places: parsePlaces(answer.text, query), answer };
  }

  async plan(query: string, candidates: Candidate[]): Promise<Answer> {
    return this.chat(PLAN_SYSTEM_PROMPT, planUserPrompt(query, candidates), planSchema(candidates.map((c) => c.id)));
  }
}
