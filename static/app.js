class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        };

        this.state = false;
        this.messages = [];
        this.intents = null;
        this.selectedOthers = false;
    }

    display(intents) {
        this.intents = intents;
        const { openButton, chatBox, sendButton } = this.args;
    
        openButton.addEventListener('click', () => {
            this.toggleState(chatBox);
    
            if (this.state) {
                if (!this.messages.some(message => message.message === "Hi there👋! Before we get started can I know your name please?")) {
                    this.sendMessage(chatBox, "Hi there👋! Before we get started, can I know your <b>name</b> please?");
                }
            }
        });
    
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));
    
        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    
        const optionsContainer = chatBox.querySelector('.chatbox__messages');
    
        // Wait for user input before displaying options
        sendButton.addEventListener('click', () => {
            const textField = chatBox.querySelector('input');
            const text1 = textField.value;
    
            if (text1 !== "") {
                if (!this.userNameEntered) {
                    let username = text1;
                    this.messages.unshift({ name: "User", message: username });
                    this.sendMessage(chatBox, `Hey ${username}, how can I help you?`, false);
                    this.userNameEntered = true;
    
                    const options = this.intents.intents.map(intent => ({ tag: intent.tag, message: intent.tag }));
                    options.push({ tag: "Others", message: "Others" });
    
                    options.forEach(option => {
                        this.sendMessage(chatBox, option.message, false, option.tag === "Others");
                    });
                } else {
                    this.onSendButton(chatBox);
                }
    
                textField.value = '';
            }
        });
    
        // Add event listener for options
        optionsContainer.addEventListener('click', (event) => {
            const selectedOption = event.target.innerText;
            this.onOptionClick(chatBox, selectedOption);
        });
    }
    
    
    
    

    onOptionClick(chatbox, selectedOption) {
        const textField = chatbox.querySelector('input');
    
        if (selectedOption === "Others") {
            // Display a response
            let responseMsg = { name: "Sam", message: "<b>Please enter your query below!</b>" };
            this.messages.push(responseMsg);
            this.updateChatText(chatbox);
    
            // Activate the textbox
            textField.removeAttribute("disabled");
            textField.placeholder = "Write a message...";
    
            // Focus on the textbox
            textField.focus();
        } else {
            const selectedIntent = this.intents.intents.find(intent => intent.tag === selectedOption);
    
            if (selectedIntent) {
                // Display user's selected option
                let msg1 = { name: "User", message: selectedOption };
                this.messages.push(msg1);
    
                // Display a random response associated with the selected option
                const randomResponse = selectedIntent.responses[Math.floor(Math.random() * selectedIntent.responses.length)];
                let msg2 = { name: "Sam", message: randomResponse };
                this.messages.push(msg2);
    
                // Update the chatbox
                this.updateChatText(chatbox);
    
                // Clear the input field
                textField.value = '';
            }
        }
    }
    
    

    sendMessage(chatbox, message, isOperator = false, isOption = false) {
        let classes = 'messages__item';
        if (isOperator) {
            classes += ' messages__item--operator';
        } else if (isOption) {
            classes += ' messages__item--option';
        } else {
            classes += ' messages__item--visitor';
        }
    
        let systemMsg = { name: "System", message: message, isOperator: isOperator, classes: classes };
        this.messages.push(systemMsg);
        this.updateChatText(chatbox);
    }

    toggleState(chatbox) {
        this.state = !this.state;
        if (this.state) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    onSendButton(chatbox) {
        const textField = chatbox.querySelector('input');
        const text1 = textField.value;
    
        if (text1 === "") {
            return;
        }
    
        if (this.selectedOthers) {
            let msg1 = { name: "User", message: text1 };
            this.messages.push(msg1);
    
            // Process the user message as needed (e.g., calling a server)
            // ...
    
            textField.value = '';
            this.selectedOthers = false;
        } else {
            if (!this.userNameEntered) {
                let username = text1;
                this.messages.push({ name: "User", message: username });
                this.sendMessage(chatbox, `Hello <b>${username}!</b> How can I help you today?`, false);
                this.userNameEntered = true;
    
                // Display initial options
                const options = this.intents.intents.map(intent => ({ tag: intent.tag, message: intent.tag }));
                options.push({ tag: "Others", message: "Others" });
    
                options.forEach(option => {
                    this.sendMessage(chatbox, option.message, false, option.tag === "Others");
                });
            } else {
                let msg1 = { name: "User", message: text1 };
                this.messages.push(msg1);
    
                fetch('/predict', {
                    method: 'POST',
                    body: JSON.stringify({ message: text1 }),
                    mode: 'cors',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                })
                    .then(r => r.json())
                    .then(r => {
                        let msg2 = { name: "Sam", message: r.answer };
                        this.messages.push(msg2);
                        this.updateChatText(chatbox);
                        textField.value = '';
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                        this.updateChatText(chatbox);
                        textField.value = '';
                    });
            }
    
            textField.value = '';
        }
    }
    
    
    

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function (item) {
            if (item.name === "System") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else if (item.name === "Sam") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });
    
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    
        // Scroll to the bottom of the chatbox
        chatmessage.scrollTop = chatmessage.scrollHeight;
    }
    
}

// Example usage
fetch('/intents.json')
    .then(response => response.json())
    .then(intents => {
        const chatbox = new Chatbox();
        chatbox.display(intents);
    })
    .catch(error => console.error('Error fetching intents:', error));
