This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Local Backend Integration

This project expects a backend Flask server running and exposing the master agent on `http://localhost:8000`.
Start the Flask backend from the repo root:

```pwsh
# Create and activate a virtualenv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r ..\requirements.txt
python ..\server.py
```

Start the Next.js dev server in the `gloser` folder:

```pwsh
cd gloser
npm install
npm run dev
```

The Next.js API route `/api/chat` will forward your chat requests to the Flask backend's `/api/master-agent/query` endpoint.
You can test integration using curl, Postman, or other tools. Example (PowerShell):

```pwsh
# Quick test (after running both servers):
Invoke-RestMethod -Uri "http://localhost:3000/api/chat" -Method Post -Body (@{ input = 'Hello' } | ConvertTo-Json) -ContentType 'application/json'
```

## Theme & UI
This app ships with a modern black theme designed for clarity and focus. Highlights:
- Gradient dark background and glassy message panels.
- Animated typing dots and message entrance animation.
- Clear Chat button to reset the conversation.
If you'd like variations (e.g., different accents or fonts), I can add a settings panel to switch themes dynamically.


## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
