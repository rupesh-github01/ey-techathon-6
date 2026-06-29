import { NextResponse } from 'next/server'

// This route proxies chat requests to the Flask server at http://localhost:8000
// and returns a normalized messages array so `useChat` can display assistant messages.

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}));

    const input = body.input || body.query || (body.messages && body.messages.slice(-1)[0]?.content);
    const sessionId = body.session_id || body.sessionId;
    
    if (!input || typeof input !== 'string') {
      return NextResponse.json({ error: 'No input provided (expected `input` or `query`).' }, { status: 400 });
    }

    // Forward the user's query to the Flask master-agent endpoint with session_id for memory
    const flaskResp = await fetch('http://localhost:8000/api/master-agent/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        query: input, 
        systemPrompt: body.systemPrompt,
        session_id: sessionId  // Pass session ID for memory continuity
      }),
    });

    // If the Flask server responded with non-JSON, we still handle it gracefully
    const data = await flaskResp.json().catch(async () => {
      console.error('Invalid JSON from backend for chat proxy');
      return { error: 'invalid_backend_response', message: 'Invalid response received from backend.' };
    });

    const assistantContent: string =
      typeof data?.result?.message === 'string'
        ? data.result.message.replace(/\s*nograph\s*$/i, '').trim()
        : data?.result?.final_answer
          ? String(data.result.final_answer)
          : data?.plan
            ? JSON.stringify(data.plan)
            : JSON.stringify(data);

    const visuals = Array.isArray(data?.result?.visuals) ? data.result.visuals : [];

    const assistantMessage = {
      id: `${Date.now()}`,
      role: 'assistant',
      content: assistantContent,
      visuals,
    } as const;

    const plan = data?.plan || (data?.result && data.result.plan) || null;
    const returnedSessionId = data?.session_id || sessionId;
    const memoryStats = data?.memory_stats || null;

    return NextResponse.json({ 
      messages: [assistantMessage], 
      visuals, 
      plan, 
      raw: data,
      session_id: returnedSessionId,
      memory_stats: memoryStats
    });
  } catch (e: any) {
    console.error('Chat proxy error:', e);
    return NextResponse.json({ error: 'server_error', message: 'Something went wrong processing the request. Please try again.' }, { status: 500 });
  }
}
