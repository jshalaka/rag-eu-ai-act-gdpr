-- Run history for the EU law RAG app. Paste into the Supabase SQL editor (or `supabase db push`).
-- Model: every signed-in team member can read all runs and rate any run; each person can only
-- insert runs as themselves, and only `rating` and `note` can be changed after a run is saved.

create table if not exists public.runs (
    run_id              text primary key,
    created_at          timestamptz not null,
    user_id             uuid not null default auth.uid() references auth.users (id),
    "user"              text not null,
    question            text not null,
    answer              text not null,
    citations           jsonb not null default '[]'::jsonb,
    retrieved_chunk_ids jsonb not null default '[]'::jsonb,
    refused             boolean not null,
    latency_ms          double precision not null,
    model               text not null,
    index_version       text not null,
    regulation_filter   text,
    article_filter      text,
    rating              smallint check (rating in (1, -1)),
    note                text
);

create index if not exists runs_created_at_idx on public.runs (created_at desc);

alter table public.runs enable row level security;

drop policy if exists "team can read runs" on public.runs;
create policy "team can read runs" on public.runs
    for select to authenticated using (true);

drop policy if exists "users insert their own runs" on public.runs;
create policy "users insert their own runs" on public.runs
    for insert to authenticated with check (user_id = auth.uid());

drop policy if exists "team can rate runs" on public.runs;
create policy "team can rate runs" on public.runs
    for update to authenticated using (true) with check (true);

-- Column-level privileges: nobody can edit a saved question or answer.
revoke all on public.runs from anon, authenticated;
grant select, insert on public.runs to authenticated;
grant update (rating, note) on public.runs to authenticated;
