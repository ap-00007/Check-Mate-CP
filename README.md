# CheckMate — NLP Course Project

A handwriting verification web app built with **React + Vite + Tailwind CSS**.  
Upload two handwriting samples and the app analyses similarity and displays an NLP-powered confidence score.

---

## 🖥️ Prerequisites

Make sure you have the following installed before running the project:

| Tool | Version | Download |
|------|---------|----------|
| **Node.js** | v18 or higher | https://nodejs.org |
| **npm** | comes with Node.js | — |

> **Check if already installed:** Open a terminal and run:
> ```
> node -v
> npm -v
> ```

---

## 🚀 Getting Started (Step-by-Step)

### 1. Download / Receive the project

Make sure you have the entire `NLP COURSE PROJECT` folder.  
> **Note:** The `node_modules` folder may or may not be included. Either way, Step 2 will handle it.

---

### 2. Open a terminal in the project folder

- **Windows:** Open the `NLP COURSE PROJECT` folder in File Explorer → Right-click inside → **"Open in Terminal"** (or PowerShell).
- **Mac/Linux:** Open Terminal and `cd` into the folder:
  ```bash
  cd path/to/NLP\ COURSE\ PROJECT
  ```

---

### 3. Install dependencies

```bash
npm install
```

This downloads all required packages into a `node_modules` folder.  
*(You only need to do this once, or when `package.json` changes.)*

---

### 4. Start the development server

```bash
npm run dev
```

You'll see output like:
```
  VITE  v8.x.x  ready in Xms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://192.168.x.x:5173/
```

---

### 5. Open in your browser

Go to 👉 **[http://localhost:5173](http://localhost:5173)**

The app will open — you can upload handwriting images and hit **Submit for Verification**.

---

## 📁 Project Structure

```
NLP COURSE PROJECT/
├── public/            # Static assets (icons, favicon)
├── src/
│   ├── components/    # UI cards (Upload, Results, Confidence, etc.)
│   ├── App.jsx        # Main app component
│   ├── main.jsx       # React entry point
│   └── index.css      # Global styles
├── package.json       # Project config & scripts
└── vite.config.js     # Vite + React + Tailwind config
```

---

## 🛠️ Available Commands

| Command | What it does |
|---------|-------------|
| `npm install` | Install all dependencies |
| `npm run dev` | Start local dev server (hot reload) |
| `npm run build` | Build for production (outputs to `dist/`) |
| `npm run preview` | Preview the production build locally |

---

## ❓ Troubleshooting

**Port already in use?**  
Vite will automatically try the next port (e.g., `5174`). Check the terminal output for the actual URL.

**`npm` not found?**  
Install Node.js from https://nodejs.org and restart your terminal.

**Blank page or errors?**  
Make sure you ran `npm install` first, then `npm run dev`.

# Check-Mate-CP
