import { TUI, Text, Input, ProcessTerminal, Box, Spacer } from "@mariozechner/pi-tui";

const withBg =
    (ansiCode: string) =>
    (text: string): string =>
        `\x1b[${ansiCode}m${text}\x1b[0m`;

const panelBg = withBg("48;2;15;23;42");
const titleBg = withBg("48;2;30;41;59");
const statusBg = withBg("48;2;24;34;50");

const terminal = new ProcessTerminal();
const tui = new TUI(terminal);

const mainBox = new Box(0, 1, panelBg);

const titleText = new Text("Chanakya - AI Assistant", 1, 0, titleBg);
const welcomeText = new Text("Enter a URL or article text. Type 'exit' to quit.", 1, 0, panelBg);
const statusText = new Text("Ready", 1, 0, statusBg);
const responseText = new Text("", 1, 0, panelBg);
const toolStatusText = new Text("", 1, 0, panelBg);
const inputField = new Input();

mainBox.addChild(titleText);
mainBox.addChild(welcomeText);
mainBox.addChild(new Spacer());
mainBox.addChild(toolStatusText);
mainBox.addChild(responseText);
mainBox.addChild(statusText);
mainBox.addChild(inputField);

tui.addChild(mainBox);
tui.setFocus(inputField);

let isProcessing = false;
let responseBuffer = "";
let toolOutputs: string[] = [];
let toolStatusBuffer = "";
let transcript = "";

function renderTranscript(activeResponse = responseBuffer): void {
    const sections: string[] = [];

    if (transcript.trim()) {
        sections.push(transcript.trimEnd());
    }

    if (activeResponse.trim()) {
        sections.push(`Assistant:\n${activeResponse.trimEnd()}`);
    }

    responseText.setText(sections.join("\n\n"));
    tui.requestRender();
}

async function sendQuery(query: string): Promise<void> {
    if (isProcessing || !query.trim()) return;

    isProcessing = true;
    responseBuffer = "";
    toolOutputs = [];
    toolStatusBuffer = "";
    if (transcript.trim()) {
        transcript += "\n\n";
    }
    transcript += `You:\n${query.trim()}`;
    statusText.setText("Processing...");
    toolStatusText.setText("");
    renderTranscript();
    tui.requestRender();

    try {
        const response = await fetch("http://127.0.0.1:8000/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) return;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split("\n");

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        handleEvent(data);
                    } catch {}
                }
            }
        }
    } catch (error) {
        renderTranscript(`Error: ${error instanceof Error ? error.message : "Failed to connect to server"}`);
        statusText.setText("Error");
        tui.requestRender();
    } finally {
        isProcessing = false;
    }
}

function handleEvent(data: { type: string; content?: string; name?: string; output?: string }): void {
    switch (data.type) {
        case "token":
            responseBuffer += data.content || "";
            renderTranscript();
            break;
        case "tool_start":
            toolStatusBuffer += `[Tool] Starting: ${data.name}\n`;
            toolStatusText.setText(toolStatusBuffer);
            tui.requestRender();
            break;
        case "tool_end":
            toolOutputs.push(`${data.name}: ${data.output}`);
            toolStatusBuffer += `[Tool] Finished: ${data.name}\n`;
            toolStatusText.setText(toolStatusBuffer);
            tui.requestRender();
            break;
        case "done":
            if (responseBuffer.trim()) {
                transcript += `\n\nAssistant:\n${responseBuffer.trimEnd()}`;
                responseBuffer = "";
                renderTranscript("");
            }
            statusText.setText("Ready");
            tui.requestRender();
            break;
    }
}

inputField.onSubmit = (text: string) => {
    if (text.toLowerCase() === "exit") {
        terminal.stop();
        return;
    }
    inputField.setValue("");
    sendQuery(text);
};

tui.start();
