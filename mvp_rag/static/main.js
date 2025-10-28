// Get references to all the important HTML elements on the page.
// This is more efficient than searching for them every time we need them.
const toast = document.getElementById("toast");
const ingestForm = document.getElementById("ingest-form");
const ingestButton = document.getElementById("ingest-button");
const ingestStatus = document.getElementById("ingest-status");
const askForm = document.getElementById("ask-form");
const askButton = document.getElementById("ask-button");
const answerBox = document.getElementById("answer");
const citationsList = document.getElementById("citations");

// A helper function to display a small notification pop-up (a "toast").
function showToast(message, isError = false) {
    toast.textContent = message;
    toast.classList.toggle("show", true); // The "show" class triggers the CSS animation.
    toast.style.background = isError ? "#b91c1c" : "#1f2937"; // Red for errors, dark gray for success.
    toast.hidden = false;
    // The toast will automatically hide after 3 seconds.
    setTimeout(() => {
        toast.classList.toggle("show", false);
    }, 3000);
}

// This function handles the submission of the file upload form.
ingestForm.addEventListener("submit", async (event) => {
    // `event.preventDefault()` stops the browser's default behavior of reloading the page on form submission.
    event.preventDefault();
    // `FormData` is a built-in browser API to easily handle form data.
    const formData = new FormData(ingestForm);
    const files = formData.getAll("files");
    // Basic validation to make sure the user has selected a file.
    if (files.length === 0 || files[0].size === 0) {
        showToast("Please select at least one PDF to upload.", true);
        return;
    }

    // Disable the button and show a status message to prevent multiple submissions.
    ingestButton.disabled = true;
    ingestStatus.textContent = "Uploading and ingesting...";

    try {
        // Use the `fetch` API to send the form data to our backend endpoint.
        const response = await fetch("/ingest", {
            method: "POST",
            body: formData,
        });
        // Parse the JSON response from the server.
        const data = await response.json();
        // If the server returned an error, throw an exception to be caught by the `catch` block.
        if (!response.ok || !data.ok) {
            throw new Error(data.error || "Ingestion failed.");
        }
        // Format the successful response to show how many chunks were added.
        const docs = (data.documents || []).map((doc) => {
            return `${doc.doc_title}: +${doc.chunks_ingested} (skipped ${doc.chunks_skipped})`;
        });
        ingestStatus.textContent = docs.length ? docs.join(" · ") : "No ingestible text found.";
        showToast("Ingestion complete.");
    } catch (error) {
        // If anything went wrong, log the error and show a user-friendly message.
        console.error(error);
        ingestStatus.textContent = "Ingestion failed.";
        showToast(error.message, true);
    } finally {
        // The `finally` block always runs, whether the request succeeded or failed.
        // This is a good place to re-enable the button and reset the form.
        ingestButton.disabled = false;
        ingestForm.reset();
    }
});

// This function handles the submission of the question form.
askForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = document.getElementById("question").value.trim();
    if (!question) {
        showToast("Please provide a question.", true);
        return;
    }

    // Disable the button and show a "Thinking..." message.
    askButton.disabled = true;
    answerBox.textContent = "Thinking...";
    citationsList.innerHTML = ""; // Clear any previous citations.

    try {
        // Send the question to the backend `/ask` endpoint.
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            // The question is sent as a JSON payload.
            body: JSON.stringify({ question }),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.error || "Unable to fetch answer.");
        }
        // Display the answer from the server.
        answerBox.textContent = data.answer || "No answer returned.";
        citationsList.innerHTML = "";
        // Dynamically create and add the citation list items.
        (data.citations || []).forEach((citation) => {
            const item = document.createElement("li");
            item.textContent = `[${citation.doc}, ${citation.page_range}]`;
            citationsList.appendChild(item);
        });
        showToast("Answer ready.");
    } catch (error) {
        console.error(error);
        answerBox.textContent = "Unable to answer.";
        showToast(error.message, true);
    } finally {
        // Re-enable the ask button once the process is complete.
        askButton.disabled = false;
    }
});
