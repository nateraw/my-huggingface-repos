import gradio as gr


def greet(name):
    return "Hello " + name + "!"


interface = gr.Interface(greet, "text", "text")

if __name__ == "__main__":
    interface.launch()
