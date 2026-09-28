// Dedicated runtime for the Open Generative UI cell. `openGenerativeUI`
// makes the provider register the `generateSandboxedUi` frontend tool and
// applies the OpenGenerativeUIMiddleware, which turns that tool call's
// arguments into `open-generative-ui` activity events rendered in a
// sandboxed iframe.
//
// open-gen-ui-advanced is not served here: its sandbox-function
// descriptors reach the model only as agent context, which this adapter
// does not fold into the prompt yet (see PARITY_NOTES.md).
//
// Reference:
// - showcase/integrations/google-adk/src/app/api/copilotkit-ogui/route.ts
// - src/agents/open_gen_ui.py

import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import {
  CopilotRuntime,
  createCopilotRuntimeHandler,
} from "@copilotkit/runtime/v2";
import { HttpAgent } from "@ag-ui/client";
import { extractForwardedHeaders } from "@/lib/header-forwarding";

const AGENT_URL = process.env.AGENT_URL || "http://localhost:8000";

export const POST = async (req: NextRequest) => {
  try {
    const headers = extractForwardedHeaders(req);

    const runtime = new CopilotRuntime({
      agents: {
        "open-gen-ui": new HttpAgent({
          url: `${AGENT_URL}/open_gen_ui`,
          headers,
        }),
      },
      openGenerativeUI: {
        agents: ["open-gen-ui"],
      },
    });

    const copilotHandler = createCopilotRuntimeHandler({
      runtime,
      basePath: "/api/copilotkit-ogui",
      mode: "single-route",
    });

    return await copilotHandler(req);
  } catch (error: unknown) {
    const e = error as { message?: string; stack?: string };
    return NextResponse.json(
      { error: e.message, stack: e.stack },
      { status: 500 },
    );
  }
};
