document.addEventListener('DOMContentLoaded', function() {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const debugLog = document.getElementById('debug-log');

    function appendMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message');
        if (sender === 'user') {
            messageElement.classList.add('user-message');
            messageElement.innerHTML = `<strong>나:</strong> ${message}`;
        } else {
            messageElement.classList.add('vac-response');
            messageElement.innerHTML = message; // Model response already formatted with HTML
        }
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight; // Auto-scroll to bottom
    }

    function appendDebugLog(messages) {
        debugLog.innerHTML = ''; // Clear previous logs
        messages.forEach(msg => {
            const logEntry = document.createElement('p');
            logEntry.textContent = msg;
            debugLog.appendChild(logEntry);
        });
        debugLog.scrollTop = debugLog.scrollHeight; // Auto-scroll
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        appendMessage('user', message);
        userInput.value = ''; // Clear input field

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });

            const data = await response.json();
            
            appendMessage('vac', data.response_html);
            appendDebugLog(data.debug_log);

        } catch (error) {
            console.error('Error sending message:', error);
            appendMessage('vac', '<p class="error-message">VAC: 메시지를 보내는 중 오류가 발생했습니다. 다시 시도해 주세요.</p>');
            appendDebugLog(['[FE 오류] 메시지 전송 중 오류 발생: ' + error.message]);
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // Initial load of debug messages (if any from app startup)
    // You might need an initial /debug_log endpoint or pass initial logs via template
    // For simplicity, this example assumes logs are only fetched after a chat message.
    // A more robust solution might fetch initial logs on page load.
});
