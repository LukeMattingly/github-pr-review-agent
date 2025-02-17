import gradio as gr
from smolagents import GradioUI

class CustomGradioUI(GradioUI):
    def launch(self, **kwargs):
        with gr.Blocks(fill_height=True) as demo:
            # Add your header and instructions at the very top
            gr.Markdown("## Welcome my Github PR Review Agent 🤖")
            gr.Markdown("Follow the instructions below to interact with the agent. Type your chat message in the box and hit enter.")


            # The rest of the UI remains the same as the original launch method
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/Alfred.png",
                ),
                resizeable=True,
                scale=1,
            )
            # If an upload folder is provided, enable the upload feature
            if self.file_upload_folder is not None:
                upload_file = gr.File(label="Upload a file")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                upload_file.change(
                    self.upload_file,
                    [upload_file, file_uploads_log],
                    [upload_status, file_uploads_log],
                )
            text_input = gr.Textbox(lines=1, label="Please provide a link to your github pull request for review.")
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(self.interact_with_agent, [stored_messages, chatbot], [chatbot])
        demo.launch(debug=True, share=True, **kwargs)

