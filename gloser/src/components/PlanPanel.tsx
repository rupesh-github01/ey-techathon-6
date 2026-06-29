'use client';

export default function PlanPanel({ plan }: { plan?: any }) {
  if (!plan) {
    return (
      <aside className="hidden md:block w-80 border-l border-white/4 px-4 py-4 plan-panel">
        <div className="text-sm text-white/60">Plan</div>
        <div className="text-xs mt-3 text-white/50">No plan available. The planner node will generate a plan when your query requires multiple agents.</div>
      </aside>
    );
  }

  return (
    <aside className="hidden md:block w-80 border-l border-white/4 px-4 py-4 plan-panel">
      <div className="flex items-center justify-between">
        <div className="text-sm text-white/80 font-semibold">Plan</div>
        <div className="text-xs text-white/50">{plan ? `${plan.length} step(s)` : ''}</div>
      </div>
      <ol className="mt-3 text-sm text-white/70 space-y-2">
        {(plan || []).map((step: any, idx: number) => (
          <li key={idx} className="flex justify-between items-center">
            <span className="font-semibold">{step.agent}</span>
            <span className="text-xs text-white/60">{step.query}</span>
          </li>
        ))}
      </ol>
    </aside>
  );
}
