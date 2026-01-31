/**
 * Paraphrase AI - Transcript Page JavaScript
 */

// Copy to clipboard functionality
function copyToClipboard() {
    const resultText = document.getElementById('resultText');
    if (!resultText) return;

    navigator.clipboard.writeText(resultText.innerText).then(() => {
        const copyBtn = document.querySelector('.copy-btn');
        if (!copyBtn) return;

        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
            Copied!
        `;
        setTimeout(() => {
            copyBtn.innerHTML = originalText;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text:', err);
    });
}
