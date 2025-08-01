
# app_launcher.py

import tkinter as tk
from tkinter import messagebox
import threading
import subprocess

def run_script_with_progress():
    def task():
        try:
            update_progress("Running...")
            subprocess.run(["python", "run_all.py"], check=True)
            update_progress("Completed")
            messagebox.showinfo("Success", "Full pipeline completed successfully!")
        except subprocess.CalledProcessError as e:
            update_progress("Failed")
            messagebox.showerror("Error", f"Pipeline failed:\n{e}")

    threading.Thread(target=task).start()

def update_progress(status):
    progress_label.config(text=f"Status: {status}")

# GUI Setup
root = tk.Tk()
root.title("Run Harness Pipeline")
root.geometry("350x160")

label = tk.Label(root, text="Click below to run full analysis", pady=10)
label.pack()

run_button = tk.Button(root, text="Run Full Pipeline", command=run_script_with_progress, height=2, width=20)
run_button.pack(pady=10)

progress_label = tk.Label(root, text="Status: Waiting")
progress_label.pack()

root.mainloop()
