const compile = document.getElementById("compile")
const upload = document.getElementById("upload")
const code = document.getElementById("code")
const credentials = document.getElementById("credentialsForm")
const loader = document.getElementById("loader");
const copyLink = document.getElementById("copyLink");

window.addEventListener("beforeunload", function () {
    navigator.sendBeacon("/logout"); // Send logout request when tab closes
});

credentials.addEventListener("submit", function(event) {
    event.preventDefault();
    
    const formData = new FormData(this);
    
    fetch("/credentials", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => alert(data.message))
    .catch(error => console.error("Error:", error));
});

compile.addEventListener("click", async () => {
    const code = document.getElementById("code").value.trim();
    const loader = document.getElementById("loader");
    loader.style.display = "block";

    if (code === "") {
        alert("Please write some code before compiling.");
        return;
    }

    try {
        const response = await fetch("/compile", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code: code }),
        });

        const result = await response.json();
        loader.style.display = "none";

        if (response.ok) {
            alert("Compilation successful!");
            document.getElementById("downloadLink").value = result.download_url;
            document.getElementById("downloadLinkContainer").style.display = "block";
        } else {
            alert("Compilation failed! Check logs.");
            console.error(result);
        }
    } catch (error) {
        alert("An error occurred during compilation.");
        console.error(error);
    }
});

copyLink.addEventListener("click", () => {
    const downloadLink = document.getElementById("downloadLink");
    downloadLink.select();
    downloadLink.setSelectionRange(0, 99999);
    navigator.clipboard.writeText(downloadLink.value);
    alert("Download link copied to clipboard!");
});


upload.addEventListener("click", function () {
    alert("Upload feature coming soon!");
});