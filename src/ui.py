import tkinter as tk
from tkinter import ttk, messagebox

# Importe für Hardwareerkennung, Spielprofil, Scoring, Assistent, API und Grafikberatung
from hardware_input import read_hardware_without_cli
from input_validation import validate_user_input
from game_profile import GameProfil
from hardware_profile import HardwareProfil
from scoring import ScoreResult
from scoring import calculate_playability_score
from assistant_agent import run_assistant
from assistant_agent import AssistantDecision
from api import search_steam_games, fetch_steam_requirements
from graphics_advisor import derive_graphics_advice
from graphics_advisor import GraphicsAdvice
from main import parse_requirements_block


class ToolTip:
    """
    Zeigt einen Tooltip mit Erklärtext beim Überfahren eines Widgets an.
    """
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self._show_tip)
        widget.bind("<Leave>", self._hide_tip)

    def _show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("TkDefaultFont", 9),
        )
        label.pack(padx=4, ipady=2)

    def _hide_tip(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class GamingAssistantApp(tk.Tk):
    """
    Hauptanwendung: Intelligenter Assistent für die Bewertung der Spielbarkeit.
    """
    def __init__(self) -> None:
        super().__init__()
        self.title("Intelligenter Assistent")
        self.geometry("950x750")

        # Zustand: Hardware und User-Input laden
        self.hardware_profile, self.user_input = read_hardware_without_cli()
        self.game_profile: GameProfil | None = None
        self.search_results: list[dict] = []

        self._build_widgets()

    def _build_widgets(self) -> None:
        """
        Erstellt das Layout der GUI mit Eingabefeldern, Schaltflächen und Ausgabebereichen.
        """
        # Eingabebereich
        frame_input = ttk.LabelFrame(self, text="Eingaben")
        frame_input.pack(fill="x", padx=10, pady=5)

        # Spielname-Eingabe
        ttk.Label(frame_input, text="Spielname:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.entry_game = ttk.Entry(frame_input, width=30)
        self.entry_game.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        btn_search = ttk.Button(frame_input, text="Spiel suchen", command=self.on_search_game)
        btn_search.grid(row=0, column=2, sticky="w", padx=5, pady=2)

        btn_analyse = ttk.Button(frame_input, text="Analyse starten", command=self.on_run_analysis)
        btn_analyse.grid(row=1, column=2, sticky="w", padx=5, pady=2)

        # Auflösung-Auswahl
        ttk.Label(frame_input, text="Auflösung:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.combo_res = ttk.Combobox(frame_input, values=["1920x1080", "2560x1440", "3840x2160"], width=27)
        self.combo_res.set("3840x2160")
        self.combo_res.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        # Ziel-FPS-Auswahl
        ttk.Label(frame_input, text="Ziel-FPS:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.combo_fps = ttk.Combobox(
            frame_input,
            values=["30", "60", "75", "120", "144", "165", "240"],
            width=10,
            state="readonly"
        )
        self.combo_fps.set("60")  # Default
        self.combo_fps.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        # Tooltip für FPS-Erklärung
        ToolTip(
            self.combo_fps,
            (
                "FPS steht für 'Frames per Seconds'.\n\n"
                "Empfehlungen für Ziel-FPS:\n\n"
                "- 30 FPS: Ausreichend für gemütliches Singeplayer-Gaming.\n"
                "- 60 FPS: Sehr flüssig, Standard-Empfehlung für die meisten Spiele.\n"
                "- 120/144 FPS: Ideal für schnelle Shooter / kompetitive Spiele,\n"
                "erfordert aber deutlich mehr GPU-/CPU-Leistung.\n"
                "- 165/240 FPS: Nur sinnvoll auf passenden Monitoren und mit sehr starker Hardware.\n\n"
                "Je höher die Ziel-FPS, desto eher solltest du Grafikdetails senken."
            ),
        )

        # Spielliste
        frame_games = ttk.LabelFrame(self, text="Gefundene Spiele")
        frame_games.pack(fill="x", padx=10, pady=5)
        self.list_games = tk.Listbox(frame_games, height=4)
        self.list_games.pack(fill="x", padx=5, pady=5)

        # Hardware vs. Spielanforderungen
        frame_compare = ttk.LabelFrame(self, text="Hardware vs. Spielanforderungen")
        frame_compare.pack(fill="x", padx=10, pady=5)
        self.text_compare = tk.Text(frame_compare, height=7)
        self.text_compare.pack(fill="x", padx=5, pady=5)

        # Ergebnisse
        frame_result = ttk.LabelFrame(self, text="Ergebnisse")
        frame_result.pack(fill="both", expand=True, padx=10, pady=5)
        self.label_score = ttk.Label(frame_result, text="Score: -/100")
        self.label_score.pack(anchor="w", padx=5, pady=2)
        self.label_level = ttk.Label(frame_result, text="Einschätzung: -")
        self.label_level.pack(anchor="w", padx=5, pady=2)
        self.text_graphics = tk.Text(frame_result, height=9)
        self.text_graphics.pack(fill="x", padx=5, pady=5)
        self.text_explanations = tk.Text(frame_result, height=9)
        self.text_explanations.pack(fill="both", expand=True, padx=5, pady=5)

    # ================================================
    # Event-Handler
    # ================================================

    def _collect_hardware_and_user(self):
        """
        Sammelt die Hardware- und User-Eingaben aus der GUI.
        """
        hardware_profile = self.hardware_profile
        user_input = self.user_input
        res = self.combo_res.get().strip()
        if res:
            hardware_profile.aufloesung = res
            user_input.resolution = res
        fps_str = self.combo_fps.get().strip()
        try:
            fps = int(fps_str)
            hardware_profile.ziel_fps = fps
            user_input.target_fps = fps
        except ValueError:
            return "Shit happens."
        return hardware_profile, user_input

    def on_search_game(self) -> None:
        """
        Sucht Spiele auf Steam basierend auf dem eingegebenen Namen.
        """
        name = self.entry_game.get().strip()
        if not name:
            messagebox.showwarning("Hinweis", "Bitte einen Spielnamen eingeben.")
            return
        try:
            results = search_steam_games(name)
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei der Spielsuche:\n{e}")
            return
        self.search_results = results
        self.list_games.delete(0, tk.END)
        for g in results:
            self.list_games.insert(tk.END, f"{g['name']} (AppID: {g['appid']})")
        if not results:
            messagebox.showinfo("Info", "Keine Spiele gefunden.")

    def on_run_analysis(self) -> None:
        """
        Führt die Analyse durch: Validierung, Spiel- und Hardwareprofil, Scoring, Grafikberatung und Assistent.
        """
        self.hardware_profile, self.user_input = self._collect_hardware_and_user()
        validation = validate_user_input(self.user_input, self.hardware_profile)
        if not validation.is_valid:
            msg = "\n".join(validation.messages)
            messagebox.showwarning("Eingabeproblem", msg)
            return
        validated_input = validation.corrected_input
        try:
            idx = self.list_games.curselection()[0]
            selected = self.search_results[idx]
        except IndexError:
            messagebox.showwarning("Hinweis", "Bitte ein Spiel aus der Liste auswählen.")
            return
        try:
            reqs = fetch_steam_requirements(selected["appid"])
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Systemanforderungen nicht laden:\n{e}")
            return
        if not reqs:
            messagebox.showerror("Fehler", "Keine Systemanforderungen gefunden.")
            return
        min_parsed = parse_requirements_block(reqs["minimum"])
        rec_parsed = parse_requirements_block(reqs["recommended"])
        self.game_profile = GameProfil(
            game_name=selected["name"],
            min_cpu=min_parsed.get("cpu", ""),
            min_gpu=min_parsed.get("gpu", ""),
            min_ram=min_parsed.get("ram_gb"),
            rec_cpu=rec_parsed.get("cpu", ""),
            rec_gpu=rec_parsed.get("gpu", ""),
            rec_ram=rec_parsed.get("ram_gb"),
        )
        self._update_compare_view()
        score_result: ScoreResult = calculate_playability_score(
            hardware_profile=self.hardware_profile,
            game_profile=self.game_profile,
            user_input=validated_input,
        )
        advice: GraphicsAdvice = derive_graphics_advice(
            score_result=score_result,
            hardware=self.hardware_profile,
            game=self.game_profile,
        )
        decision: AssistantDecision = run_assistant(score_result)
        self._update_result_view(score_result, advice, decision)

    # ================================================
    # Hilfsfunktionen fuer UI
    # ================================================

    def _update_compare_view(self) -> None:
        """
        Aktualisiert die Ansicht für den Vergleich von Hardware und Spielanforderungen.
        """
        hp = self.hardware_profile
        gp = self.game_profile
        self.text_compare.delete("1.0", tk.END)
        self.text_compare.insert(tk.END, f" CPU: {hp.cpu}\n")
        self.text_compare.insert(tk.END, f" GPU: {hp.gpu}\n")
        self.text_compare.insert(tk.END, f" RAM: {hp.ram} GB\n")
        self.text_compare.insert(tk.END, f" Auflösung/Ziel-FPS: {hp.aufloesung} @ {hp.ziel_fps}\n\n")
        self.text_compare.insert(tk.END, f"{gp.game_name} - Anforderungen:\n")
        self.text_compare.insert(tk.END, f" Minimum: {gp.min_cpu}, {gp.min_gpu}, {gp.min_ram} GB RAM\n")
        self.text_compare.insert(tk.END, f" Empfohlen: {gp.rec_cpu}, {gp.rec_gpu}, {gp.rec_ram} GB RAM\n")

    def _update_result_view(self, score_result: ScoreResult, advice: GraphicsAdvice, decision: AssistantDecision) -> None:
        """
        Aktualisiert die Ergebnisansicht mit Score, Einschätzung, Grafikempfehlungen und Erklärungen.
        """
        self.label_score.config(text=f"Score: {decision.final_score}/100")
        self.label_level.config(text=f"Einschätzung: {decision.final_label}")
        self.text_graphics.delete("1.0", tk.END)
        self.text_graphics.insert(tk.END,"Empfohlene Grafikeinstellungen:\n")
        self.text_graphics.insert(tk.END, f"- Preset: {advice.preset}\n")
        self.text_graphics.insert(tk.END, f"- Raytracing: {advice.raytracing}\n")
        self.text_graphics.insert(tk.END, f"- Upscaling: {advice.upscaling}\n")
        self.text_graphics.insert(tk.END, f"- V-Sync: {advice.vsync}\n")
        if advice.resolution_hint:
            self.text_graphics.insert(tk.END, f"- Auflösung: {advice.resolution_hint}\n")
        if advice.explanations:
            self.text_graphics.insert(tk.END, "\nErklärungen:\n")
            for expl in advice.explanations:
                self.text_graphics.insert(tk.END, f"- {expl}\n")
        self.text_explanations.delete("1.0", tk.END)
        self.text_explanations.insert(tk.END, "Begründung des Assistenten:\n")
        for line in decision.explanations:
            self.text_explanations.insert(tk.END, f"- {line}\n")
        self.text_explanations.insert(tk.END, "\nEmpfehlungen:\n")
        for rec in decision.recommendations:
            self.text_explanations.insert(tk.END, f"- {rec}\n")
        if decision.ai_label:
            self.text_explanations.insert(tk.END, "\nKI-basierte Einschätzung:\n")
            self.text_explanations.insert(tk.END, f"- {decision.ai_label}\n")
            if decision.ai_agreement:
                self.text_explanations.insert(tk.END, f"- {decision.ai_agreement}\n")
        self.text_graphics.yview_moveto(0)
        self.text_explanations.yview_moveto(0)


if __name__ == "__main__":
    app = GamingAssistantApp()
    app.mainloop()
