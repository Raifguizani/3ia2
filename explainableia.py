import os
import ollama
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import logging
import re
import json
from typing import Dict, List, Tuple, Union, Optional
from concurrent.futures import ThreadPoolExecutor
import functools
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import shap
from wordcloud import WordCloud
import base64
from io import BytesIO
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XAIFeatures:
    """Classe pour les fonctionnalités d'Explainable AI."""
    
    @staticmethod
    def create_skill_match_heatmap(cv_skills: List[str], job_skills: List[str], similarity_matrix: np.ndarray) -> str:
        """
        Crée une heatmap de similarité entre les compétences du CV et de l'offre d'emploi.
        
        Args:
            cv_skills: Liste des compétences du CV
            job_skills: Liste des compétences de l'offre d'emploi
            similarity_matrix: Matrice de similarité entre les compétences
            
        Returns:
            Image encodée en base64 pour affichage HTML
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                   xticklabels=job_skills, yticklabels=cv_skills)
        plt.title("Correspondance entre compétences du CV et exigences du poste")
        plt.xlabel("Compétences requises")
        plt.ylabel("Compétences du candidat")
        plt.tight_layout()
        
        # Sauvegarder l'image dans un buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Encoder en base64 pour l'affichage HTML
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
    
    @staticmethod
    def compute_skill_importance(cv_text: str, skills: List[str]) -> Dict[str, float]:
        """
        Calcule l'importance de chaque compétence dans le CV en utilisant TF-IDF.
        
        Args:
            cv_text: Texte du CV
            skills: Liste des compétences à évaluer
            
        Returns:
            Dictionnaire avec les compétences et leur score d'importance
        """
        # Si pas de compétences, retourner un dictionnaire vide
        if not skills:
            return {}
            
        # Création d'un vectoriseur TF-IDF
        vectorizer = TfidfVectorizer(vocabulary=skills, lowercase=True)
        
        try:
            # Transformation du texte en vecteurs TF-IDF
            tfidf_matrix = vectorizer.fit_transform([cv_text])
            
            # Récupération des scores pour chaque compétence
            feature_names = vectorizer.get_feature_names_out()
            skill_scores = {}
            
            # Récupération des scores de chaque compétence
            for i, skill in enumerate(feature_names):
                score = tfidf_matrix[0, i]
                if skill in skills:
                    skill_scores[skill] = float(score)
                    
            # Pour les compétences non trouvées dans le vectoriseur
            for skill in skills:
                if skill not in skill_scores:
                    # Recherche par expression régulière pour les compétences composées
                    pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
                    matches = pattern.findall(cv_text)
                    skill_scores[skill] = len(matches) / len(cv_text.split()) if matches else 0.0
                    
            return skill_scores
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de l'importance des compétences: {str(e)}")
            return {skill: 0.0 for skill in skills}
    
    @staticmethod
    def create_skill_importance_chart(skill_importance: Dict[str, float]) -> str:
        """
        Crée un graphique horizontal montrant l'importance de chaque compétence.
        
        Args:
            skill_importance: Dictionnaire des compétences et leur importance
            
        Returns:
            Image encodée en base64 pour affichage HTML
        """
        if not skill_importance:
            # Retourner une image vide si pas de données
            plt.figure(figsize=(10, 2))
            plt.text(0.5, 0.5, "Aucune donnée d'importance des compétences disponible", 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            return img_str
            
        # Trier les compétences par importance
        sorted_skills = sorted(skill_importance.items(), key=lambda x: x[1], reverse=True)
        skills = [item[0] for item in sorted_skills]
        scores = [item[1] for item in sorted_skills]
        
        # Limiter à 15 compétences pour la lisibilité
        if len(skills) > 15:
            skills = skills[:15]
            scores = scores[:15]
            
        # Créer un graphique à barres horizontales
        plt.figure(figsize=(10, max(6, len(skills) * 0.4)))
        bars = plt.barh(skills, scores, color='skyblue')
        plt.xlabel('Score d\'importance')
        plt.title('Importance des compétences dans le CV')
        plt.tight_layout()
        
        # Ajouter les valeurs à côté des barres
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                    ha='left', va='center')
        
        # Sauvegarder l'image dans un buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Encoder en base64 pour l'affichage HTML
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
    
    @staticmethod
    def create_skill_wordcloud(text: str, title: str) -> str:
        """
        Crée un nuage de mots à partir du texte.
        
        Args:
            text: Texte à analyser
            title: Titre du nuage de mots
            
        Returns:
            Image encodée en base64 pour affichage HTML
        """
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            max_words=150, contour_width=3, contour_color='steelblue')
        
        # Générer le nuage de mots
        wordcloud.generate(text)
        
        # Afficher le nuage de mots
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        
        # Sauvegarder l'image dans un buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Encoder en base64 pour l'affichage HTML
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str

    @staticmethod
    def create_score_comparison_chart(scores: Dict[str, float]) -> str:
        """
        Crée un graphique radar des scores de correspondance.
        
        Args:
            scores: Dictionnaire des catégories et leurs scores
            
        Returns:
            Image encodée en base64 pour affichage HTML
        """
        categories = list(scores.keys())
        values = list(scores.values())
        
        # Nombre de catégories
        N = len(categories)
        if N < 3:
            # Pas assez de catégories pour un graphique radar
            # Utiliser un graphique à barres à la place
            plt.figure(figsize=(10, 6))
            plt.bar(categories, values, color='skyblue')
            plt.ylim(0, 100)
            plt.title('Scores de correspondance par catégorie')
            plt.ylabel('Score (%)')
            
            for i, v in enumerate(values):
                plt.text(i, v + 2, f"{v}%", ha='center')
                
            plt.tight_layout()
            
            # Sauvegarder l'image dans un buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Encoder en base64 pour l'affichage HTML
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
        
        # Conversion en coordonnées angulaires
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Fermer le graphique
        
        # Ajouter la valeur de la première catégorie à la fin pour fermer le graphique
        values_radar = values.copy()
        values_radar += values_radar[:1]
        
        # Créer le graphique radar
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Tracer le graphique radar
        ax.plot(angles, values_radar, 'o-', linewidth=2, label='Scores')
        ax.fill(angles, values_radar, alpha=0.25)
        
        # Définir les étiquettes et le titre
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 100)
        ax.set_title('Scores de correspondance par catégorie', size=15)
        
        # Ajouter des cercles concentriques et des étiquettes de valeur
        ax.set_rticks([20, 40, 60, 80, 100])
        ax.set_yticklabels([f"{x}%" for x in [20, 40, 60, 80, 100]])
        ax.grid(True)
        
        # Ajouter des étiquettes de valeur sur chaque point
        for i, (angle, value) in enumerate(zip(angles[:-1], values)):
            ax.text(angle, value + 10, f"{value}%", 
                   horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        # Sauvegarder l'image dans un buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Encoder en base64 pour l'affichage HTML
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
    
    @staticmethod
    def create_feature_importance_visualization(cv_text: str, job_text: str, top_features: int = 10) -> str:
        """
        Alternative à LIME: créer une visualisation des mots importants dans la correspondance.
        
        Args:
            cv_text: Texte du CV
            job_text: Texte de l'offre d'emploi
            top_features: Nombre de caractéristiques à afficher
            
        Returns:
            Image encodée en base64 pour affichage HTML
        """
        try:
            # Tokenization basique (peut être améliorée)
            def preprocess_text(text):
                # Convertir en minuscules et diviser en mots
                words = re.findall(r'\b\w+\b', text.lower())
                # Filtrer les mots courts (stopwords simplifiés)
                return [w for w in words if len(w) > 3]
            
            cv_words = preprocess_text(cv_text)
            job_words = preprocess_text(job_text)
            
            # Trouver les mots communs (intersection)
            common_words = set(cv_words) & set(job_words)
            
            # Calculer l'importance basée sur la fréquence dans les deux documents
            word_importance = {}
            
            for word in common_words:
                cv_freq = cv_words.count(word) / len(cv_words)
                job_freq = job_words.count(word) / len(job_words)
                # Score d'importance combiné
                word_importance[word] = (cv_freq * job_freq) ** 0.5 * 100
            
            # Trouver des mots uniques au CV qui pourraient être pertinents
            unique_cv_words = set(cv_words) - set(job_words)
            for word in unique_cv_words:
                if word in job_text.lower():  # Le mot est présent dans le job mais pas comme mot complet
                    importance = cv_words.count(word) / len(cv_words) * 30
                    word_importance[word] = importance
            
            # Trouver les mots présents uniquement dans l'offre (manquants dans le CV)
            missing_words = set(job_words) - set(cv_words)
            for word in missing_words:
                if job_words.count(word) > 1:  # Filtrer pour ne garder que les mots répétés
                    importance = -job_words.count(word) / len(job_words) * 50
                    word_importance[word] = importance
            
            # Trier et limiter aux top_features dans chaque catégorie
            positive_words = sorted([(w, s) for w, s in word_importance.items() if s > 0], 
                                   key=lambda x: x[1], reverse=True)[:top_features]
            
            negative_words = sorted([(w, s) for w, s in word_importance.items() if s < 0], 
                                   key=lambda x: x[1])[:top_features]
            
            # Créer un graphique à barres horizontales
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Mots positifs (présents dans le CV et pertinents)
            words = [x[0] for x in positive_words]
            scores = [x[1] for x in positive_words]
            pos_bars = ax.barh(range(len(words)), scores, color='green', alpha=0.6)
            
            # Mots négatifs (absents du CV mais importants dans l'offre)
            words.extend([x[0] for x in negative_words])
            scores.extend([x[1] for x in negative_words])
            
            # Ajuster la taille du graphique en fonction du nombre de mots
            plt.figure(figsize=(10, max(6, len(words) * 0.3)))
            
            # Créer le graphique final
            plt.figure(figsize=(10, len(words)*0.4 + 2))
            bars = plt.barh(words, scores, color=['green' if s > 0 else 'red' for s in scores])
            
            # Ajouter des informations
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Impact sur la correspondance')
            plt.title('Mots qui impactent la correspondance CV-Offre')
            
            # Ajouter des annotations
            for bar in bars:
                width = bar.get_width()
                color = 'black'
                alignment = 'left' if width < 0 else 'right'
                position = width - 5 if width >= 0 else width + 5
                plt.text(position, bar.get_y() + bar.get_height()/2, 
                        f'{abs(width):.1f}', 
                        ha=alignment, va='center', color=color)
            
            plt.tight_layout()
            
            # Sauvegarder l'image dans un buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Encoder en base64 pour l'affichage HTML
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la visualisation d'importance: {str(e)}")
            
            # Créer une image d'erreur
            plt.figure(figsize=(10, 2))
            plt.text(0.5, 0.5, f"Erreur lors de la création de la visualisation: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center', color='red')
            plt.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str

    @staticmethod
    def create_term_impact_visualization(cv_terms: Dict[str, float], job_terms: Dict[str, float]) -> str:
        """
        Crée une visualisation comparative des termes importants du CV vs l'offre d'emploi.
        
        Args:
            cv_terms: Dictionnaire des termes du CV et leur importance
            job_terms: Dictionnaire des termes de l'offre d'emploi et leur importance
            
        Returns:
            Image encodée en base64 pour affichage HTML
        """
        try:
            # Extraire les termes communs
            common_terms = set(cv_terms.keys()) & set(job_terms.keys())
            
            # Créer un DataFrame pour faciliter la visualisation
            data = []
            
            for term in common_terms:
                cv_score = cv_terms.get(term, 0)
                job_score = job_terms.get(term, 0)
                match_score = min(cv_score, job_score) / max(cv_score, job_score) * 100
                data.append({
                    'term': term,
                    'cv_score': cv_score,
                    'job_score': job_score,
                    'match_score': match_score
                })
            
            # Ajouter quelques termes importants non communs
            cv_only = set(cv_terms.keys()) - common_terms
            top_cv_only = sorted([(t, cv_terms[t]) for t in cv_only], key=lambda x: x[1], reverse=True)[:5]
            
            for term, score in top_cv_only:
                data.append({
                    'term': term + ' (CV)',
                    'cv_score': score,
                    'job_score': 0,
                    'match_score': 0
                })
                
            job_only = set(job_terms.keys()) - common_terms
            top_job_only = sorted([(t, job_terms[t]) for t in job_only], key=lambda x: x[1], reverse=True)[:5]
            
            for term, score in top_job_only:
                data.append({
                    'term': term + ' (Offre)',
                    'cv_score': 0,
                    'job_score': score,
                    'match_score': 0
                })
            
            # Trier par score de correspondance
            data = sorted(data, key=lambda x: x['match_score'], reverse=True)
            
            # Limiter à 15 termes pour la lisibilité
            if len(data) > 15:
                data = data[:15]
                
            # Créer la visualisation
            plt.figure(figsize=(12, 10))
            
            # Subplot pour le score de correspondance
            plt.subplot(2, 1, 1)
            terms = [d['term'] for d in data]
            match_scores = [d['match_score'] for d in data]
            
            match_bars = plt.barh(terms, match_scores, color='purple', alpha=0.7)
            plt.xlabel('Score de correspondance (%)')
            plt.title('Correspondance des termes entre CV et Offre d\'emploi')
            plt.xlim(0, 110)
            
            # Ajouter les valeurs
            for bar in match_bars:
                width = bar.get_width()
                if width > 0:
                    plt.text(width + 2, bar.get_y() + bar.get_height()/2, 
                            f'{width:.1f}%', ha='left', va='center')
            
            # Subplot pour les scores individuels
            plt.subplot(2, 1, 2)
            x = np.arange(len(terms))
            width = 0.35
            
            cv_values = [d['cv_score'] for d in data]
            job_values = [d['job_score'] for d in data]
            
            plt.barh(x - width/2, cv_values, width, label='Importance dans le CV', color='blue', alpha=0.6)
            plt.barh(x + width/2, job_values, width, label='Importance dans l\'offre', color='red', alpha=0.6)
            
            plt.yticks(x, terms)
            plt.xlabel('Score d\'importance')
            plt.title('Importance des termes dans le CV vs Offre d\'emploi')
            plt.legend()
            
            plt.tight_layout()
            
            # Sauvegarder l'image dans un buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Encoder en base64 pour l'affichage HTML
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la visualisation d'impact des termes: {str(e)}")
            
            # Créer une image d'erreur
            plt.figure(figsize=(10, 2))
            plt.text(0.5, 0.5, f"Erreur lors de la création de la visualisation: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center', color='red')
            plt.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
    @staticmethod
    def shap_summary_plot(features: pd.DataFrame, model, feature_names: List[str]) -> str:
        """
        Crée un graphique SHAP pour visualiser l'importance des caractéristiques.
        
        Args:
            features: DataFrame des caractéristiques
            model: Modèle pour les prédictions
            feature_names: Noms des caractéristiques
            
        Returns:
            Image encodée en base64 pour affichage HTML
        """
        try:
            # Créer un explainer SHAP
            explainer = shap.Explainer(model, features)
            shap_values = explainer(features)
            
            # Créer un graphique SHAP
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
            plt.title("Importance des caractéristiques pour la correspondance CV-offre d'emploi")
            plt.tight_layout()
            
            # Sauvegarder l'image dans un buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Encoder en base64 pour l'affichage HTML
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du graphique SHAP: {str(e)}")
            
            # Créer une image d'erreur
            plt.figure(figsize=(10, 2))
            plt.text(0.5, 0.5, f"Erreur lors de la création du graphique SHAP: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center', color='red')
            plt.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
    @staticmethod
    def generate_html_report(cv_info: Dict, job_requirements: Dict, match_details: Dict, 
                           xai_visuals: Dict) -> str:
        """
        Génère un rapport HTML expliquant la correspondance entre le CV et l'offre d'emploi.
        
        Args:
            cv_info: Informations extraites du CV
            job_requirements: Exigences du poste
            match_details: Détails de la correspondance
            xai_visuals: Visuels d'explainabilité (images en base64)
            
        Returns:
            Rapport HTML
        """
        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rapport d'analyse expliquée - {cv_info['personal_info'].get('name', 'Candidat')}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .score-card {{
                    display: inline-block;
                    width: 120px;
                    height: 120px;
                    margin: 10px;
                    text-align: center;
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .score-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .score-label {{
                    font-size: 14px;
                }}
                .high {{
                    color: #27ae60;
                }}
                .medium {{
                    color: #f39c12;
                }}
                .low {{
                    color: #e74c3c;
                }}
                .skill-match {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 10px;
                }}
                .skill-name {{
                    width: 200px;
                }}
                .skill-bar {{
                    height: 20px;
                    background: #e1e1e1;
                    border-radius: 10px;
                    width: 300px;
                    margin: 0 10px;
                }}
                .skill-fill {{
                    height: 100%;
                    border-radius: 10px;
                    background: #3498db;
                }}
                .skill-value {{
                    width: 50px;
                    text-align: right;
                }}
                .missing {{
                    background: #ffecec;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 5px;
                }}
                .strength {{
                    background: #e8f8f5;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 5px;
                }}
                .recommendation {{
                    background: #fff8e1;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 5px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px auto;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f9fa;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 10px;
                    font-size: 12px;
                    color: #7f8c8d;
                }}
                .toggle-button {{
                    background: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 15px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 10px 0;
                }}
                .hidden {{
                    display: none;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Rapport d'analyse expliquée</h1>
                <h2>{cv_info['personal_info'].get('name', 'Candidat')} - {job_requirements.get('job_title', 'Poste')}</h2>
                <p>Date d'analyse: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
            </div>
            
            <div class="section">
                <h2>Résumé de la correspondance</h2>
                <p>Ce rapport explique en détail pourquoi le candidat a obtenu un score de correspondance global de <strong>{match_details['overall_match']}%</strong> pour ce poste.</p>
                
                <div style="text-align: center;">
        """
        
        # Ajouter les scores avec code couleur
        score_classes = {
            'overall_match': 'Correspondance globale',
            'technical_skills_match': 'Compétences techniques',
            'education_match': 'Formation',
            'experience_match': 'Expérience',
            'languages_match': 'Langues',
            'soft_skills_match': 'Soft skills'
        }
        
        for key, label in score_classes.items():
            if key in match_details:
                score = match_details[key]
                color_class = 'high' if score >= 70 else ('medium' if score >= 40 else 'low')
                html += f"""
                <div class="score-card">
                    <div class="score-label">{label}</div>
                    <div class="score-value {color_class}">{score}%</div>
                </div>
                """
        
        # Ajouter le graphique radar des scores
        if 'score_radar' in xai_visuals:
            html += f"""
                </div>
                <h3>Visualisation des scores par catégorie</h3>
                <p>Ce graphique radar montre les scores de correspondance pour chaque catégorie évaluée :</p>
                <img src="data:image/png;base64,{xai_visuals['score_radar']}" alt="Graphique radar des scores">
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Explication de la correspondance technique</h2>
                <p>Cette section explique en détail pourquoi le candidat a obtenu ce score pour les compétences techniques.</p>
        """
        
        # Ajouter la heatmap de correspondance des compétences
        if 'skill_heatmap' in xai_visuals:
            html += f"""
                <h3>Correspondance entre compétences du CV et exigences du poste</h3>
                <p>Cette visualisation montre comment chaque compétence du candidat correspond aux exigences du poste :</p>
                <img src="data:image/png;base64,{xai_visuals['skill_heatmap']}" alt="Heatmap de correspondance des compétences">
                
                <button class="toggle-button" onclick="document.getElementById('compDetails').classList.toggle('hidden')">
                    Voir les détails des compétences comparées
                </button>
                
                <div id="compDetails" class="hidden">
                    <h4>Compétences techniques requises :</h4>
                    <ul>
            """
            
            for skill in job_requirements.get('technical_skills', []):
                html += f"<li>{skill}</li>"
                
            html += """
                    </ul>
                    
                    <h4>Compétences techniques du candidat :</h4>
                    <ul>
            """
            
            for skill in cv_info['skills'].get('technical_skills', []):
                html += f"<li>{skill}</li>"
                
            html += """
                    </ul>
                </div>
            """
        
        # Ajouter le graphique d'importance des compétences
        if 'skill_importance' in xai_visuals:
            html += f"""
                <h3>Importance des compétences dans le CV</h3>
                <p>Ce graphique montre l'importance relative des compétences techniques dans le CV du candidat, 
                basée sur leur fréquence et leur contexte :</p>
                <img src="data:image/png;base64,{xai_visuals['skill_importance']}" alt="Importance des compétences">
            """
        
        # Ajouter les nuages de mots
        if 'cv_wordcloud' in xai_visuals:
            html += f"""
                <h3>Nuage de mots du CV</h3>
                <p>Ce nuage de mots met en évidence les termes les plus fréquents dans le CV du candidat :</p>
                <img src="data:image/png;base64,{xai_visuals['cv_wordcloud']}" alt="Nuage de mots du CV">
            """
            
        if 'job_wordcloud' in xai_visuals:
            html += f"""
                <h3>Nuage de mots de l'offre d'emploi</h3>
                <p>Ce nuage de mots met en évidence les termes les plus fréquents dans l'offre d'emploi :</p>
                <img src="data:image/png;base64,{xai_visuals['job_wordcloud']}" alt="Nuage de mots de l'offre d'emploi">
            """
            
        html += """
            </div>
            
            <div class="section">
                <h2>Analyse des termes d'influence</h2>
                <p>Cette analyse met en évidence les termes qui influencent positivement ou négativement la correspondance avec l'offre d'emploi :</p>
        """
        
        # Ajouter l'explication par termes d'influence (remplace LIME)
        if 'feature_importance' in xai_visuals:
            html += f"""
                <img src="data:image/png;base64,{xai_visuals['feature_importance']}" alt="Importance des caractéristiques">
                <p>Les termes en vert contribuent positivement à la correspondance, tandis que les termes en rouge représentent des compétences manquantes ou sous-représentées dans le CV.</p>
            """
            
        # Ajouter la comparaison d'impact des termes
        if 'term_impact' in xai_visuals:
            html += f"""
                <h3>Comparaison de l'importance des termes</h3>
                <p>Cette visualisation compare l'importance relative des termes clés dans le CV et dans l'offre d'emploi :</p>
                <img src="data:image/png;base64,{xai_visuals['term_impact']}" alt="Impact des termes">
                <p>Un score de correspondance élevé indique que le terme a une importance similaire dans le CV et l'offre d'emploi.</p>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Importance des caractéristiques (SHAP)</h2>
                <p>Cette visualisation montre comment différentes caractéristiques du CV influencent le score de correspondance global :</p>
        """
        
        # Ajouter le graphique SHAP
        if 'shap_analysis' in xai_visuals:
            html += f"""
                <img src="data:image/png;base64,{xai_visuals['shap_analysis']}" alt="Analyse SHAP">
                <p>Les caractéristiques en haut du graphique ont plus d'impact sur le score final. 
                Les points rouges indiquent des valeurs élevées pour la caractéristique, les points bleus des valeurs faibles.</p>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Points forts et axes d'amélioration</h2>
        """
        
        # Ajouter les points forts
        html += """
                <h3>Points forts du candidat</h3>
                <p>Ces éléments du profil du candidat correspondent particulièrement bien aux exigences du poste :</p>
        """
        
        if match_details.get('strengths'):
            for strength in match_details['strengths']:
                html += f'<div class="strength">✓ {strength}</div>'
        else:
            html += '<p>Aucun point fort spécifique identifié.</p>'
            
        # Ajouter les compétences manquantes
        html += """
                <h3>Compétences manquantes</h3>
                <p>Ces compétences requises pour le poste n'ont pas été clairement identifiées dans le CV du candidat :</p>
        """
        
        if match_details.get('missing_skills'):
            for skill in match_details['missing_skills']:
                html += f'<div class="missing">✗ {skill}</div>'
        else:
            html += '<p>Aucune compétence manquante majeure identifiée.</p>'
            
        # Ajouter les recommandations
        html += """
                <h3>Recommandations pour améliorer la correspondance</h3>
                <p>Voici des suggestions pour améliorer l'adéquation du profil avec ce type de poste :</p>
        """
        
        if match_details.get('recommendations'):
            for rec in match_details['recommendations']:
                html += f'<div class="recommendation">→ {rec}</div>'
        else:
            html += '<p>Aucune recommandation spécifique.</p>'
            
        html += """
            </div>
            
            <div class="section">
                <h2>Évaluation finale</h2>
                <div style="white-space: pre-line;">
        """
        
        # Ajouter l'évaluation finale
        if 'final_assessment' in match_details:
            html += match_details['final_assessment']
        else:
            html += "Aucune évaluation finale disponible."
            
        html += """
                </div>
            </div>
            
            <div class="footer">
                <p>Ce rapport d'analyse expliquée a été généré automatiquement à l'aide de techniques d'intelligence artificielle explicable (XAI).</p>
                <p>© {datetime.now().year} - CV Parser & Matcher avec XAI</p>
            </div>
            
            <script>
                // Script pour les boutons toggle
                document.addEventListener('DOMContentLoaded', function() {
                    var toggleButtons = document.querySelectorAll('.toggle-button');
                    toggleButtons.forEach(function(button) {
                        button.addEventListener('click', function() {
                            var targetId = this.getAttribute('data-target');
                            var targetElement = document.getElementById(targetId);
                            if (targetElement) {
                                targetElement.classList.toggle('hidden');
                            }
                        });
                    });
                });
            </script>
        </body>
        </html>
        """
        
        return html


class CVParser:
    def __init__(self, embedding_model_name: str = "BAAI/bge-m3", llm_model_name: str = "llama3", smtp_config: Dict = None):
        logger.info(f"Initializing with embedding model: {embedding_model_name}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")
            
        self.llm_model = llm_model_name
        
        try:
            logger.info(f"Pulling LLM model: {llm_model_name}")
            ollama.pull(self.llm_model)
        except Exception as e:
            logger.warning(f"Failed to pull LLM model, will attempt to use anyway: {str(e)}")
        
        self.smtp_config = smtp_config or {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "gastonishere1000@gmail.com",
        "sender_password": "ijif bzrq gyom mqbl",
        "google_credentials_file": "credentials.json"
    }
        
        # Initialiser la classe XAI
        self.xai = XAIFeatures()


    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if not text.strip():
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise RuntimeError(f"Error extracting text from PDF: {str(e)}")


    def get_embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
            
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def _extract_json_from_llm_response(self, response_text: str) -> Dict:
        json_pattern = r'({[\s\S]*})'
        json_match = re.search(json_pattern, response_text)
        
        if json_match:
            json_str = json_match.group(1)
            json_str = re.sub(r'```(?:json)?\s*', '', json_str)
            json_str = re.sub(r'```\s*$', '', json_str)
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse cleaned JSON")
                try:
                    # Try to fix common JSON formatting issues
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON after cleanup attempt")
                    return {}
        else:
            logger.warning("No JSON pattern found in LLM response")
            return {}

    def extract_personal_info(self, cv_text: str) -> Dict:
        info = {
            "name": None,
            "email": None,
            "phone": None,
            "address": None,
            "linkedin": None,
            "github": None,
            "website": None
        }
        
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        email_matches = re.findall(email_pattern, cv_text)
        if email_matches:
            info["email"] = email_matches[0]
        
        phone_patterns = [
            r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'(?:\+\d{1,3}[-.\s]?)?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}[-.\s]?\d{2}',
            r'(?:\+\d{1,3}[-.\s]?)?\d{10}'
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, cv_text)
            if phone_match:
                info["phone"] = phone_match.group(0)
                break
        
        linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+'
        linkedin_match = re.search(linkedin_pattern, cv_text)
        if linkedin_match:
            info["linkedin"] = linkedin_match.group(0)
            if not info["linkedin"].startswith('http'):
                info["linkedin"] = "https://" + info["linkedin"]
                
        github_pattern = r'(?:https?://)?(?:www\.)?github\.com/[\w-]+'
        github_match = re.search(github_pattern, cv_text)
        if github_match:
            info["github"] = github_match.group(0)
            if not info["github"].startswith('http'):
                info["github"] = "https://" + info["github"]
                
        website_pattern = r'(?:https?://)?(?:www\.)?[\w-]+\.(?:com|org|net|io|dev)(?:/[\w-]*)*'
        website_matches = re.findall(website_pattern, cv_text)
        if website_matches:
            for match in website_matches:
                if "linkedin.com" not in match and "github.com" not in match:
                    info["website"] = match
                    if not info["website"].startswith('http'):
                        info["website"] = "https://" + info["website"]
                    break
        
        try:
            prompt = f"""
            Extrais uniquement les informations suivantes du texte du CV ci-dessous :
            - Nom complet de la personne
            - Adresse complète (si disponible)
            - Liens LinkedIn, GitHub et site web personnel (si disponibles)
            
            Réponds STRICTEMENT au format JSON exact ci-dessous, sans aucun autre texte:
            {{
                "name": "Prénom Nom",
                "address": "Adresse complète ou null si non disponible",
                "linkedin": "URL LinkedIn ou null si non disponible",
                "github": "URL GitHub ou null si non disponible",
                "website": "URL site web personnel ou null si non disponible"
            }}
            
            Texte du CV (extrait):
            ---
            {cv_text[:3000]}
            ---
            """
            
            response = ollama.chat(model=self.llm_model, messages=[
                {"role": "system", "content": "Tu es un extracteur de données précises. Réponds UNIQUEMENT en JSON valide."},
                {"role": "user", "content": prompt}
            ])
            
            llm_info = self._extract_json_from_llm_response(response["message"]["content"])
            if llm_info:
                if llm_info.get("name"):
                    info["name"] = llm_info.get("name")
                if llm_info.get("address") and llm_info.get("address") != "null":
                    info["address"] = llm_info.get("address")
                if llm_info.get("linkedin") and llm_info.get("linkedin") != "null":
                    info["linkedin"] = llm_info.get("linkedin")
                if llm_info.get("github") and llm_info.get("github") != "null":
                    info["github"] = llm_info.get("github")
                if llm_info.get("website") and llm_info.get("website") != "null":
                    info["website"] = llm_info.get("website")
            
        except Exception as e:
            logger.error(f"Error extracting personal info with LLM: {str(e)}")
        
        return info

    def _llm_extraction(self, prompt: str, system_message: str) -> Dict:
        try:
            response = ollama.chat(model=self.llm_model, messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ])
            
            return self._extract_json_from_llm_response(response["message"]["content"])
        except Exception as e:
            logger.error(f"Error in LLM extraction: {str(e)}")
            return {}

    def extract_education(self, cv_text: str) -> List[Dict]:
        prompt = f"""
        Extrais UNIQUEMENT les informations d'éducation formelle (diplômes académiques) du CV ci-dessous.
        Ne pas inclure les certifications professionnelles, celles-ci seront traitées séparément.

        Pour chaque formation académique, identifie :
        - Le diplôme ou titre obtenu (ex: Master, Licence, BTS, etc.)
        - L'établissement/école/université
        - La période (dates de début et fin, ou en cours)
        - Le domaine d'étude ou spécialisation
        - La localisation (ville, pays) si mentionnée
        - La note/mention si mentionnée
        
        Réponds UNIQUEMENT au format JSON:
        {{
            "education": [
                {{
                    "degree": "Nom du diplôme",
                    "institution": "Nom de l'établissement",
                    "period": "Période",
                    "field": "Domaine d'étude",
                    "location": "Ville, Pays ou null si non disponible",
                    "grade": "Mention ou note ou null si non disponible"
                }},
                ...
            ]
        }}
        
        Texte du CV:
        ---
        {cv_text}
        ---
        """
        
        system_message = "Tu es un expert d'extraction de formations académiques. Réponds UNIQUEMENT en JSON valide."
        result = self._llm_extraction(prompt, system_message)
        return result.get("education", [])

    def extract_experience(self, cv_text: str) -> List[Dict]:
        prompt = f"""
        Extrais TOUTES les informations d'expérience professionnelle du CV ci-dessous.
        Pour chaque expérience, identifie :
        - Le poste/titre exact
        - L'entreprise/organisation
        - La période précise (dates de début et fin, ou en cours)
        - La localisation (ville, pays) si mentionnée
        - Les responsabilités ou réalisations principales (liste détaillée)
        - Le type de contrat si mentionné (CDI, CDD, stage, freelance, etc.)
        
        Réponds UNIQUEMENT au format JSON:
        {{
            "experience": [
                {{
                    "title": "Titre du poste",
                    "company": "Nom de l'entreprise",
                    "period": "Période",
                    "location": "Ville, Pays ou null si non disponible",
                    "responsibilities": ["Responsabilité 1", "Responsabilité 2", ...],
                    "contract_type": "Type de contrat ou null si non spécifié"
                }},
                ...
            ]
        }}
        
        Texte du CV:
        ---
        {cv_text}
        ---
        """
        
        system_message = "Tu es un expert d'extraction d'expériences professionnelles. Réponds UNIQUEMENT en JSON valide."
        result = self._llm_extraction(prompt, system_message)
        return result.get("experience", [])

    def extract_skills(self, cv_text: str) -> Dict[str, List[str]]:
        prompt = f"""
        Extrais et catégorise TOUTES les compétences du CV ci-dessous.
        Identifie les compétences dans ces catégories précises :
        - Compétences techniques (technologies, langages, frameworks, outils, méthodologies)
        - Langues (avec niveau si mentionné)
        - Compétences non techniques / soft skills

        Pour les soft skills, assure-toi d'ajouter des espaces entre les mots pour une meilleure lisibilité.
        Par exemple "Développement de solides compétences en communication" au lieu de "Développementdesolidescompétencesencommunication".
        
        Réponds UNIQUEMENT au format JSON:
        {{
            "technical_skills": ["Compétence 1", "Compétence 2", ...],
            "languages": ["Langue 1 (niveau)", "Langue 2 (niveau)", ...],
            "soft_skills": ["Soft skill 1", "Soft skill 2", ...]
        }}
        
        Texte du CV:
        ---
        {cv_text}
        ---
        """
        
        system_message = "Tu es un expert d'extraction de compétences. Réponds UNIQUEMENT en JSON valide."
        result = self._llm_extraction(prompt, system_message)
        
        # Ensure all skill arrays contain strings only and fix formatting
        skills_result = {
            "technical_skills": [],
            "languages": [],
            "soft_skills": []
        }
        
        # Handle technical skills
        tech_skills = result.get("technical_skills", [])
        for skill in tech_skills:
            if isinstance(skill, str):
                skills_result["technical_skills"].append(skill)
            elif isinstance(skill, dict):
                # Handle case where skill is a dict
                skills_result["technical_skills"].append(str(skill))
        
        # Handle languages
        languages = result.get("languages", [])
        for lang in languages:
            if isinstance(lang, str):
                skills_result["languages"].append(lang)
            elif isinstance(lang, dict):
                # If language is returned as a dict (e.g. {"language": "English", "level": "Fluent"})
                if "language" in lang and "level" in lang:
                    skills_result["languages"].append(f"{lang['language']} ({lang['level']})")
                else:
                    # Just convert the dict to a string representation
                    skills_result["languages"].append(str(lang))
        
        # Handle soft skills - ensure proper spacing
        soft_skills = result.get("soft_skills", [])
        for skill in soft_skills:
            if isinstance(skill, str):
                # Add spaces if words are run together
                processed_skill = re.sub(r'([a-z])([A-Z])', r'\1 \2', skill)
                # Add spaces after periods if missing
                processed_skill = re.sub(r'\.([A-Z])', r'. \1', processed_skill)
                skills_result["soft_skills"].append(processed_skill)
            elif isinstance(skill, dict):
                # Handle case where skill is a dict
                skills_result["soft_skills"].append(str(skill))
                
        return skills_result
        
    def extract_certifications(self, cv_text: str) -> List[Dict]:
        prompt = f"""
        Extrais TOUTES les certifications et formations complémentaires du CV ci-dessous.
        Attention à bien inclure les certifications techniques comme NVIDIA, IBM, Microsoft, AWS, etc.
        
        Pour chaque certification, identifie :
        - Le nom/titre complet de la certification
        - L'organisme émetteur (NVIDIA, IBM, Microsoft, etc.)
        - La date d'obtention ou d'expiration
        - Le domaine (par exemple "Intelligence Artificielle", "Cloud Computing", etc.)
        - L'ID ou référence si mentionné
        
        Réponds UNIQUEMENT au format JSON:
        {{
            "certifications": [
                {{
                    "name": "Nom de la certification",
                    "issuer": "Organisme émetteur",
                    "date": "Date d'obtention/expiration ou null si non disponible",
                    "field": "Domaine ou null si non disponible",
                    "id": "Identifiant ou null si non disponible"
                }},
                ...
            ]
        }}
        
        Texte du CV:
        ---
        {cv_text}
        ---
        """
        
        system_message = "Tu es un expert d'extraction de certifications. Réponds UNIQUEMENT en JSON valide."
        result = self._llm_extraction(prompt, system_message)
        return result.get("certifications", [])
        
    def extract_projects(self, cv_text: str) -> List[Dict]:
        prompt = f"""
        Extrais TOUS les projets personnels ou professionnels mentionnés dans le CV ci-dessous.
        Pour chaque projet, identifie :
        - Le titre/nom du projet
        - La description/objectif principal
        - Les technologies ou compétences utilisées
        - La période ou durée
        - Les résultats/impacts si mentionnés
        - Le lien ou référence si mentionné
        
        Réponds UNIQUEMENT au format JSON:
        {{
            "projects": [
                {{
                    "title": "Nom du projet",
                    "description": "Description du projet",
                    "technologies": ["Technologie 1", "Technologie 2", ...],
                    "period": "Période ou durée",
                    "results": "Résultats ou impacts ou null si non disponible",
                    "link": "Lien vers le projet ou null si non disponible"
                }},
                ...
            ]
        }}
        
        Texte du CV:
        ---
        {cv_text}
        ---
        """
        
        system_message = "Tu es un expert d'extraction de projets. Réponds UNIQUEMENT en JSON valide."
        result = self._llm_extraction(prompt, system_message)
        return result.get("projects", [])

    def _clean_soft_skills(self, soft_skills: List[str]) -> List[str]:
        """Helper function to clean and format soft skills"""
        cleaned_skills = []
        for skill in soft_skills:
            # Add spaces between camelCase words
            skill = re.sub(r'([a-z])([A-Z])', r'\1 \2', skill)
            # Add spaces after periods if missing
            skill = re.sub(r'\.([A-Z])', r'. \1', skill)
            # Add spaces after commas if missing
            skill = re.sub(r',([A-Za-z])', r', \1', skill)
            # Fix run-together words by adding spaces (this is a heuristic approach)
            # Look for transitions between lowercase and uppercase letters
            skill = re.sub(r'([a-z])([A-Z])', r'\1 \2', skill)
            cleaned_skills.append(skill)
        return cleaned_skills


    def generate_cv_summary(self, cv_info: Dict) -> str:
        try:
            cv_text = f"""
            Nom: {cv_info['personal_info'].get('name', 'Non spécifié')}
            Email: {cv_info['personal_info'].get('email', 'Non spécifié')}
            Téléphone: {cv_info['personal_info'].get('phone', 'Non spécifié')}
            
            Expérience professionnelle (top 3):
            {self._format_experiences(cv_info['experience'][:3])}
            
            Formation (top 2):
            {self._format_education(cv_info['education'][:2])}
            
            Compétences techniques clés:
            {', '.join(cv_info['skills'].get('technical_skills', [])[:10])}
            
            Langues:
            {', '.join(cv_info['skills'].get('languages', []))}
            
            Certifications:
            {self._format_certifications(cv_info['certifications'][:3])}
            
            Projets notables:
            {self._format_projects(cv_info['projects'][:2])}
            """
            
            prompt = f"""
            Génère un résumé professionnel impactant (environ 150-200 mots) basé sur les informations de CV ci-dessous.
            Le résumé doit:
            - Être à la première personne (je suis...)
            - Mettre en avant l'expérience pertinente et le positionnement professionnel
            - Mentionner les compétences techniques distinctives
            - Inclure un ou deux accomplissements significatifs
            - Être dynamique et convaincant pour un recruteur
            
            Informations du CV:
            ---
            {cv_text}
            ---
            """
            
            response = ollama.chat(model=self.llm_model, messages=[
                {"role": "system", "content": "Tu es un expert en rédaction de profils professionnels percutants."},
                {"role": "user", "content": prompt}
            ])
            
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating CV summary: {str(e)}")
            return "Impossible de générer un résumé pour ce CV."
    
    def _format_experiences(self, experiences: List[Dict]) -> str:
        if not experiences:
            return "Aucune expérience listée"
        
        result = ""
        for exp in experiences:
            result += f"- {exp.get('title', 'Poste')} chez {exp.get('company', 'Entreprise')} ({exp.get('period', 'Période non spécifiée')})\n"
            if exp.get('location'):
                result += f"  Lieu: {exp.get('location')}\n"
            resp = exp.get('responsibilities', [])
            if resp and len(resp) > 0:
                result += f"  Principales responsabilités: {resp[0]}"
                if len(resp) > 1:
                    result += f", {resp[1]}"
                result += "\n"
        
        return result
    
    def _format_education(self, education: List[Dict]) -> str:
        if not education:
            return "Aucune formation listée"
        
        result = ""
        for edu in education:
            result += f"- {edu.get('degree', 'Diplôme')} en {edu.get('field', 'domaine non spécifié')} - {edu.get('institution', 'Institution')} ({edu.get('period', 'Période non spécifiée')})\n"
            if edu.get('location'):
                result += f"  Lieu: {edu.get('location')}\n"
            if edu.get('grade'):
                result += f"  Mention: {edu.get('grade')}\n"
        
        return result
        
    def _format_certifications(self, certifications: List[Dict]) -> str:
        if not certifications:
            return "Aucune certification listée"
        
        result = ""
        for cert in certifications:
            result += f"- {cert.get('name', 'Certification')} - {cert.get('issuer', 'Émetteur')}"
            if cert.get('date'):
                result += f" ({cert.get('date')})"
            result += "\n"
            if cert.get('field'):
                result += f"  Domaine: {cert.get('field')}\n"
            if cert.get('id'):
                result += f"  ID: {cert.get('id')}\n"
        
        return result
        
    def _format_projects(self, projects: List[Dict]) -> str:
        if not projects:
            return "Aucun projet listé"
        
        result = ""
        for proj in projects:
            result += f"- {proj.get('title', 'Projet')} ({proj.get('period', 'Période non spécifiée')})\n"
            if proj.get('description'):
                result += f"  {proj.get('description')[:100]}...\n"
            if proj.get('technologies'):
                tech_list = proj.get('technologies', [])
                # Check if technologies is a list of strings
                if tech_list and isinstance(tech_list[0], str):
                    result += f"  Technologies: {', '.join(tech_list[:5])}\n"
                else:
                    # Handle case where technologies might be a list of dicts
                    tech_strings = []
                    for tech in tech_list[:5]:
                        if isinstance(tech, str):
                            tech_strings.append(tech)
                        elif isinstance(tech, dict):
                            tech_strings.append(str(tech))
                    result += f"  Technologies: {', '.join(tech_strings)}\n"
        
        return result

    def parse_cv(self, pdf_path: str) -> Dict:
        logger.info(f"Extracting text from CV: {pdf_path}")
        cv_text = self.extract_text_from_pdf(pdf_path)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            personal_info_future = executor.submit(self.extract_personal_info, cv_text)
            education_future = executor.submit(self.extract_education, cv_text)
            experience_future = executor.submit(self.extract_experience, cv_text)
            skills_future = executor.submit(self.extract_skills, cv_text)
            certifications_future = executor.submit(self.extract_certifications, cv_text)
            projects_future = executor.submit(self.extract_projects, cv_text)
            
            personal_info = personal_info_future.result()
            education = education_future.result()
            experience = experience_future.result()
            skills = skills_future.result()
            certifications = certifications_future.result()
            projects = projects_future.result()
        
        # Clean soft skills to ensure proper spacing
        if 'soft_skills' in skills:
            skills['soft_skills'] = self._clean_soft_skills(skills['soft_skills'])
        
        cv_info = {
            "personal_info": personal_info,
            "education": education,
            "experience": experience,
            "skills": skills,
            "certifications": certifications,
            "projects": projects,
            "raw_text": cv_text
        }
        
        logger.info("Generating CV summary")
        cv_info["summary"] = self.generate_cv_summary(cv_info)
        
        return cv_info

    def generate_job_embedding(self, job_description: str) -> np.ndarray:
        if not job_description.strip():
            logger.warning("Empty job description provided for embedding")
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
            
        try:
            return self.embedding_model.encode(job_description)
        except Exception as e:
            logger.error(f"Error generating job embedding: {str(e)}")
            raise RuntimeError(f"Error generating job embedding: {str(e)}")

    def calculate_similarity(self, cv_embedding: np.ndarray, job_embedding: np.ndarray) -> float:
        try:
            # Calculate cosine similarity
            return np.dot(cv_embedding, job_embedding) / (np.linalg.norm(cv_embedding) * np.linalg.norm(job_embedding))
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def extract_job_requirements(self, job_description: str) -> Dict:
        prompt = f"""
        Extrais les exigences principales de cette offre d'emploi:
        
        1. Titre du poste
        2. Compétences techniques requises (liste)
        3. Expérience minimale requise (en années)
        4. Formation/diplômes requis
        5. Compétences non techniques (soft skills) demandées
        6. Langues requises
        
        Réponds UNIQUEMENT au format JSON:
        {{
            "job_title": "Titre du poste",
            "technical_skills": ["Compétence 1", "Compétence 2", ...],
            "experience_years": "Nombre d'années ou description (ex: '3 ans', 'Junior', 'Senior')",
            "education": ["Diplôme 1", "Diplôme 2", ...],
            "soft_skills": ["Soft skill 1", "Soft skill 2", ...],
            "languages": ["Langue 1 (niveau)", "Langue 2 (niveau)", ...]
        }}
        
        Offre d'emploi:
        ---
        {job_description}
        ---
        """
        
        system_message = "Tu es un expert d'analyse d'offres d'emploi. Réponds UNIQUEMENT en JSON valide."
        return self._llm_extraction(prompt, system_message)

    def calculate_skills_similarity(self, cv_skills: List[str], job_skills: List[str]) -> Tuple[float, np.ndarray]:
        """
        Calcule la similarité entre les compétences du CV et celles de l'offre d'emploi.
        
        Args:
            cv_skills: Liste des compétences techniques du CV
            job_skills: Liste des compétences techniques requises pour le poste
            
        Returns:
            Un tuple contenant (score_global, matrice_de_similarité)
        """
        if not cv_skills or not job_skills:
            logger.warning("Liste de compétences vide fournie pour le calcul de similarité")
            return 0.0, np.zeros((1, 1))
            
        try:
            # Créer des embeddings pour chaque compétence
            cv_embeddings = [self.embedding_model.encode(skill) for skill in cv_skills]
            job_embeddings = [self.embedding_model.encode(skill) for skill in job_skills]
            
            # Calculer la matrice de similarité
            similarity_matrix = np.zeros((len(cv_skills), len(job_skills)))
            
            for i, cv_emb in enumerate(cv_embeddings):
                for j, job_emb in enumerate(job_embeddings):
                    similarity_matrix[i, j] = np.dot(cv_emb, job_emb) / (np.linalg.norm(cv_emb) * np.linalg.norm(job_emb))
            
            # Calculer le score global avec une approche greedy
            max_similarity = 0.0
            matched_indices = set()
            
            # Pour chaque compétence du job, trouver la meilleure correspondance dans le CV
            for j in range(len(job_skills)):
                best_sim = 0.0
                best_idx = -1
                
                for i in range(len(cv_skills)):
                    if i not in matched_indices and similarity_matrix[i, j] > best_sim:
                        best_sim = similarity_matrix[i, j]
                        best_idx = i
                
                if best_idx != -1:
                    matched_indices.add(best_idx)
                    max_similarity += best_sim
            
            # Calculer le score final normalisé
            if len(job_skills) > 0:
                max_similarity /= len(job_skills)
            else:
                max_similarity = 0.0
                
            return max_similarity, similarity_matrix
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la similarité des compétences: {str(e)}")
            return 0.0, np.zeros((len(cv_skills) if cv_skills else 1, len(job_skills) if job_skills else 1))
    
    def _compute_text_similarity_for_explanation(self, text1: str, text2: str) -> float:
        """
        Calcule la similarité entre deux textes.
        
        Args:
            text1: Premier texte
            text2: Deuxième texte
            
        Returns:
            Score de similarité
        """
        try:
            emb1 = self.embedding_model.encode(text1)
            emb2 = self.embedding_model.encode(text2)
            
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        except Exception as e:
            logger.error(f"Erreur lors du calcul de similarité: {str(e)}")
            return 0.0

    def extract_important_terms(self, text: str, min_length: int = 4) -> Dict[str, float]:
        """
        Extrait les termes importants d'un texte avec leurs scores.
        
        Args:
            text: Texte à analyser
            min_length: Longueur minimale des termes à considérer
            
        Returns:
            Dictionnaire des termes et leur importance
        """
        try:
            # Extraire les termes et calculer leur fréquence
            words = re.findall(r'\b\w+\b', text.lower())
            words = [w for w in words if len(w) >= min_length]
            
            # Filtrer les mots très communs (stopwords simplifiés)
            stopwords = {'dans', 'pour', 'avec', 'cette', 'plus', 'vous', 'nous', 'votre', 'notre', 'être', 'avoir'}
            words = [w for w in words if w not in stopwords]
            
            # Calculer la fréquence
            term_freq = {}
            for word in words:
                if word in term_freq:
                    term_freq[word] += 1
                else:
                    term_freq[word] = 1
            
            # Normaliser les scores
            total_words = len(words)
            if total_words > 0:
                term_importance = {word: count / total_words * 100 for word, count in term_freq.items()}
            else:
                term_importance = {}
                
            # Trier et limiter
            sorted_terms = sorted(term_importance.items(), key=lambda x: x[1], reverse=True)
            top_terms = dict(sorted_terms[:50])  # Limiter aux 50 termes les plus importants
            
            return top_terms
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des termes importants: {str(e)}")
            return {}

    def generate_features_for_shap(self, cv_info: Dict, job_requirements: Dict) -> Tuple[pd.DataFrame, List[str]]:
        """
        Génère des caractéristiques pour l'analyse SHAP.
        
        Args:
            cv_info: Informations du CV
            job_requirements: Exigences du poste
            
        Returns:
            DataFrame des caractéristiques et liste des noms de caractéristiques
        """
        try:
            # Initialiser les caractéristiques avec des valeurs par défaut
            features = {
                "technical_skills_match": 0.0,
                "education_match": 0.0,
                "experience_match": 0.0,
                "languages_match": 0.0,
                "soft_skills_match": 0.0,
                "domain_relevance": 0.0
            }
            
            # Calcul de correspondance des compétences techniques
            cv_skills = cv_info['skills'].get('technical_skills', [])
            job_skills = job_requirements.get('technical_skills', [])
            
            if cv_skills and job_skills:
                tech_match, _ = self.calculate_skills_similarity(cv_skills, job_skills)
                features["technical_skills_match"] = tech_match
            
            # Calcul de correspondance des formations
            cv_degrees = [edu.get('degree', '').lower() for edu in cv_info['education']]
            cv_fields = [edu.get('field', '').lower() for edu in cv_info['education']]
            job_education = [edu.lower() for edu in job_requirements.get('education', [])]
            
            education_match = 0.0
            if job_education and cv_degrees:
                for job_edu in job_education:
                    for i, cv_degree in enumerate(cv_degrees):
                        if (job_edu in cv_degree) or (cv_fields[i] and job_edu in cv_fields[i]):
                            education_match += 1
                            break
                
                education_match = min(1.0, education_match / len(job_education))
            
            features["education_match"] = education_match
            
            # Calcul de correspondance des langues
            cv_languages = [lang.lower() for lang in cv_info['skills'].get('languages', [])]
            job_languages = [lang.lower() for lang in job_requirements.get('languages', [])]
            
            language_match = 0.0
            if job_languages and cv_languages:
                for job_lang in job_languages:
                    for cv_lang in cv_languages:
                        if job_lang.split()[0] in cv_lang:  # Comparer juste le nom de la langue
                            language_match += 1
                            break
                
                language_match = min(1.0, language_match / len(job_languages))
            
            features["languages_match"] = language_match
            
            # Calcul de correspondance des soft skills
            cv_soft = cv_info['skills'].get('soft_skills', [])
            job_soft = job_requirements.get('soft_skills', [])
            
            if cv_soft and job_soft:
                soft_match, _ = self.calculate_skills_similarity(cv_soft, job_soft)
                features["soft_skills_match"] = soft_match
            
            # Calcul de correspondance de l'expérience
            experience_match = 0.0
            job_exp_req = job_requirements.get('experience_years', '').lower()
            cv_experience = cv_info['experience']
            
            if job_exp_req and cv_experience:
                # Extraire le nombre d'années requis
                years_pattern = r'(\d+)[\s-]*ans?'
                years_match = re.search(years_pattern, job_exp_req)
                required_years = int(years_match.group(1)) if years_match else 0
                
                # Estimer les années d'expérience du CV
                total_years = 0
                for exp in cv_experience:
                    period = exp.get('period', '')
                    years_pattern = r'(\d{4})\s*[-–]\s*(\d{4}|présent|actuel|aujourd\'hui)'
                    period_match = re.search(years_pattern, period, re.IGNORECASE)
                    
                    if period_match:
                        start = int(period_match.group(1))
                        end_str = period_match.group(2).lower()
                        
                        if end_str in ['présent', 'actuel', 'aujourd\'hui']:
                            end = datetime.now().year
                        else:
                            end = int(end_str)
                            
                        total_years += (end - start)
                
                # Calculer le score de correspondance
                if required_years > 0:
                    experience_match = min(1.0, total_years / required_years)
                else:
                    # Si aucune année spécifique n'est requise, considérer simplement s'il y a de l'expérience
                    experience_match = 1.0 if cv_experience else 0.0
            
            features["experience_match"] = experience_match
            
            # Calculer la pertinence du domaine
            cv_text = " ".join([
                " ".join([exp.get('title', '') for exp in cv_info['experience']]),
                " ".join([edu.get('field', '') for edu in cv_info['education']]),
                " ".join(cv_info['skills'].get('technical_skills', []))
            ])
            
            job_text = " ".join([
                job_requirements.get('job_title', ''),
                " ".join(job_requirements.get('technical_skills', [])),
                " ".join(job_requirements.get('education', []))
            ])
            
            domain_relevance = self._compute_text_similarity_for_explanation(cv_text, job_text)
            features["domain_relevance"] = domain_relevance
            
            # Créer un DataFrame pour SHAP
            features_df = pd.DataFrame([features])
            feature_names = list(features.keys())
            
            return features_df, feature_names
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des caractéristiques pour SHAP: {str(e)}")
            features_df = pd.DataFrame([{
                "technical_skills_match": 0.0,
                "education_match": 0.0,
                "experience_match": 0.0,
                "languages_match": 0.0,
                "soft_skills_match": 0.0,
                "domain_relevance": 0.0
            }])
            feature_names = ["technical_skills_match", "education_match", "experience_match", 
                           "languages_match", "soft_skills_match", "domain_relevance"]
            return features_df, feature_names

    def generate_match_model(self, features_df: pd.DataFrame) -> callable:
        """
        Génère un modèle simple pour les prédictions de correspondance basé sur les caractéristiques.
        
        Args:
            features_df: DataFrame des caractéristiques
            
        Returns:
            Fonction de prédiction
        """
        # Définir des poids pour chaque caractéristique
        weights = {
            "technical_skills_match": 0.35,
            "domain_relevance": 0.25,
            "experience_match": 0.15,
            "education_match": 0.10,
            "languages_match": 0.10,
            "soft_skills_match": 0.05
        }
        
        def predict(X):
            predictions = np.zeros(len(X))
            
            for i, row in X.iterrows():
                score = 0.0
                for feature, weight in weights.items():
                    if feature in row:
                        score += row[feature] * weight
                
                predictions[i] = score
                
            return predictions
            
        return predict

    def match_cv_with_job(self, cv_info: Dict, job_description: str) -> Dict:
        logger.info("Matching CV with job description")
    
        # Extract key job requirements
        job_requirements = self.extract_job_requirements(job_description)
    
        # Generate embeddings for both CV and job description
        cv_text = cv_info["raw_text"]
        cv_embedding = self.get_embedding(cv_text)
        job_embedding = self.generate_job_embedding(job_description)
    
        # Calculate initial similarity score using embeddings
        initial_similarity = self.calculate_similarity(cv_embedding, job_embedding)
    
        # Initialize detailed matching scores
        match_details = {
            "overall_match": 0,  # Will be calculated later with emphasis on technical skills
            "technical_skills_match": 0,
            "education_match": 0,
            "experience_match": 0,
            "languages_match": 0,
            "soft_skills_match": 0,
            "missing_skills": [],
            "strengths": [],
            "recommendations": []
        }
    
        # Perform detailed analysis using LLM
        prompt = f"""
    Analyse la correspondance entre ce CV et cette offre d'emploi de manière TRÈS CRITIQUE. 
    Le secteur/domaine doit correspondre avant tout.
    
    CV (Extrait):
    ```
    Nom: {cv_info['personal_info'].get('name', 'Non spécifié')}
    
    Compétences techniques:
    {', '.join(cv_info['skills'].get('technical_skills', []))}
    
    Formation:
    {self._format_education(cv_info['education'])}
    
    Expérience:
    {self._format_experiences(cv_info['experience'])}
    
    Langues:
    {', '.join(cv_info['skills'].get('languages', []))}
    
    Soft skills:
    {', '.join(cv_info['skills'].get('soft_skills', []))}
    ```
    
    Exigences du poste:
    ```
    Titre: {job_requirements.get('job_title', 'Non spécifié')}
    
    Compétences techniques requises:
    {', '.join(job_requirements.get('technical_skills', []))}
    
    Formation requise:
    {', '.join(job_requirements.get('education', []))}
    
    Expérience requise:
    {job_requirements.get('experience_years', 'Non spécifiée')}
    
    Langues requises:
    {', '.join(job_requirements.get('languages', []))}
    
    Soft skills requis:
    {', '.join(job_requirements.get('soft_skills', []))}
    ```
    
    IMPORTANT: Si le domaine du CV (data science, développement web, marketing, etc.) est différent 
    du domaine du poste (ingénierie mécanique, finance, etc.), le score de correspondance technique 
    doit être TRÈS BAS (inférieur à 20%).
    
    Réponds STRICTEMENT au format JSON suivant:
{{
    "technical_skills_match": score entre 0 et 100,
    "education_match": score entre 0 et 100,
    "experience_match": score entre 0 et 100,
    "languages_match": score entre 0 et 100,
    "soft_skills_match": score entre 0 et 100,
    "domain_match": score entre 0 et 100 indiquant si le domaine d'expertise du CV correspond au domaine du poste,
    "missing_skills": ["compétence manquante 1", "compétence manquante 2", ...],
    "strengths": ["point fort 1", "point fort 2", ...],
    "recommendations": ["recommandation 1", "recommandation 2", ...]
}}

    
    Les scores doivent représenter le pourcentage de correspondance du CV avec les exigences du poste.
    Les compétences manquantes sont les compétences requises qui ne sont pas présentes dans le CV.
    Les points forts sont les aspects du CV qui correspondent particulièrement bien aux exigences.
    Les recommandations sont des conseils pour améliorer le CV pour ce poste spécifique.
        """
    
        try:
            response = ollama.chat(model=self.llm_model, messages=[
                {"role": "system", "content": "Tu es un expert en recrutement capable d'analyser précisément la correspondance entre un CV et une offre d'emploi. Tu es TRÈS exigeant sur la correspondance du domaine d'expertise. Réponds UNIQUEMENT en JSON valide."},
                {"role": "user", "content": prompt}
            ])
        
            detailed_match = self._extract_json_from_llm_response(response["message"]["content"])
        
            # Update match details with LLM analysis
            if detailed_match:
                for key in detailed_match:
                    if key in match_details:
                        match_details[key] = detailed_match[key]
            
                # Calculate overall match with heavy emphasis on technical skills and domain match
                technical_weight = 0.6  # 60% weight for technical skills
                domain_weight = 0.2     # 20% weight for domain match
                other_weight = 0.05     # 5% weight for each other category
            
                domain_match = detailed_match.get("domain_match", 0)
            
                overall_score = (
                    technical_weight * match_details["technical_skills_match"] +
                    domain_weight * domain_match +
                    other_weight * match_details["education_match"] +
                    other_weight * match_details["experience_match"] +
                    other_weight * match_details["languages_match"] +
                    other_weight * match_details["soft_skills_match"]
                )
            
                # If domain match is very low (less than 30%), cap the overall score
                if domain_match < 30:
                    overall_score = min(overall_score, 40)
            
                # If technical skills match is very low (less than 30%), cap the overall score
                if match_details["technical_skills_match"] < 30:
                    overall_score = min(overall_score, 35)
                
                match_details["overall_match"] = round(overall_score, 2)
                match_details["domain_match"] = domain_match
        except Exception as e:
            logger.error(f"Error in LLM detailed matching: {str(e)}")
            # Fallback: calculate a basic score with heavy emphasis on technical skills
            match_details["overall_match"] = round(initial_similarity * 100 * 0.3, 2)  # Much lower weight for embedding similarity
    
        # Generate final match assessment
        match_details["job_requirements"] = job_requirements
        match_details["final_assessment"] = self.generate_match_assessment(cv_info, job_requirements, match_details)
        
        # Generate XAI visualizations
        match_details["xai_visuals"] = self.generate_xai_visualizations(cv_info, job_requirements, match_details)
        
        return match_details
    
    def generate_xai_visualizations(self, cv_info: Dict, job_requirements: Dict, match_details: Dict) -> Dict:
        """
        Génère les visualisations XAI pour expliquer la correspondance.
        
        Args:
            cv_info: Informations du CV
            job_requirements: Exigences du poste
            match_details: Détails de la correspondance
            
        Returns:
            Dictionnaire des visualisations XAI encodées en base64
        """
        xai_visuals = {}
        
        try:
            # 1. Générer le graphique radar des scores
            score_categories = {
                "Technique": match_details["technical_skills_match"],
                "Domaine": match_details.get("domain_match", 0),
                "Formation": match_details["education_match"],
                "Expérience": match_details["experience_match"],
                "Langues": match_details["languages_match"],
                "Soft skills": match_details["soft_skills_match"]
            }
            xai_visuals["score_radar"] = self.xai.create_score_comparison_chart(score_categories)
            
            # 2. Générer la heatmap de correspondance des compétences
            cv_skills = cv_info['skills'].get('technical_skills', [])[:10]  # Limiter pour la lisibilité
            job_skills = job_requirements.get('technical_skills', [])[:10]  # Limiter pour la lisibilité
            
            if cv_skills and job_skills:
                _, similarity_matrix = self.calculate_skills_similarity(cv_skills, job_skills)
                xai_visuals["skill_heatmap"] = self.xai.create_skill_match_heatmap(cv_skills, job_skills, similarity_matrix)
            
            # 3. Calculer et visualiser l'importance des compétences dans le CV
            if cv_skills:
                cv_text = cv_info["raw_text"]
                skill_importance = self.xai.compute_skill_importance(cv_text, cv_skills)
                xai_visuals["skill_importance"] = self.xai.create_skill_importance_chart(skill_importance)
            
            # 4. Générer des nuages de mots pour CV et job
            xai_visuals["cv_wordcloud"] = self.xai.create_skill_wordcloud(
                " ".join(cv_info['skills'].get('technical_skills', [])), 
                "Compétences techniques du candidat"
            )
            
            xai_visuals["job_wordcloud"] = self.xai.create_skill_wordcloud(
                " ".join(job_requirements.get('technical_skills', [])), 
                "Compétences techniques requises pour le poste"
            )
            
            # 5. Analyse des termes importants (alternative à LIME)
            cv_terms = self.extract_important_terms(cv_info["raw_text"])
            job_terms = self.extract_important_terms(
                job_requirements.get("job_title", "") + " " + 
                " ".join(job_requirements.get("technical_skills", []))
            )
            
            # Visualisation de l'importance des termes
            xai_visuals["feature_importance"] = self.xai.create_feature_importance_visualization(
                cv_info["raw_text"],
                job_requirements.get("job_title", "") + " " + " ".join(job_requirements.get("technical_skills", []))
            )
            
            # Comparaison d'impact des termes
            xai_visuals["term_impact"] = self.xai.create_term_impact_visualization(cv_terms, job_terms)
            
            # 6. Analyse SHAP
            features_df, feature_names = self.generate_features_for_shap(cv_info, job_requirements)
            model = self.generate_match_model(features_df)
            xai_visuals["shap_analysis"] = self.xai.shap_summary_plot(features_df, model, feature_names)
            
            return xai_visuals
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des visualisations XAI: {str(e)}")
            return xai_visuals
    
    def generate_match_assessment(self, cv_info: Dict, job_requirements: Dict, match_details: Dict) -> str:
        prompt = f"""
    Génère une évaluation finale (environ 200-250 mots) sur la correspondance entre ce candidat et ce poste.
    L'évaluation doit:
    - Commencer par une conclusion générale sur l'adéquation du candidat au poste
    - Mettre en évidence les principales forces du candidat par rapport aux exigences
    - Mentionner les lacunes ou compétences manquantes importantes
    - Mentionner EXPLICITEMENT si le domaine d'expertise du candidat ne correspond pas au domaine du poste
    - Donner une recommandation (embaucher, entretien, rejeter) basée sur les scores
    - Suggérer 1-2 questions spécifiques à poser lors d'un éventuel entretien
    
    Scores de correspondance:
    - Global: {match_details['overall_match']}%
    - Compétences techniques: {match_details['technical_skills_match']}%
    - Correspondance du domaine: {match_details.get('domain_match', 'Non évalué')}%
    - Formation: {match_details['education_match']}%
    - Expérience: {match_details['experience_match']}%
    - Langues: {match_details['languages_match']}%
    - Soft skills: {match_details['soft_skills_match']}%
    
    Compétences manquantes: {', '.join(match_details['missing_skills'])}
    
    Candidat: {cv_info['personal_info'].get('name', 'Candidat')}
    Poste: {job_requirements.get('job_title', 'Poste')}
    
    IMPORTANT: Si le score de correspondance technique est inférieur à 40% OU si la correspondance de domaine est inférieure à 30%, la recommandation devrait généralement être "rejeter", sauf circonstances exceptionnelles.
    """
    
        try:
            response = ollama.chat(model=self.llm_model, messages=[
                {"role": "system", "content": "Tu es un expert en recrutement qui fournit des évaluations objectives et détaillées de l'adéquation entre candidats et postes, en étant particulièrement vigilant sur la correspondance du domaine d'expertise."},
                {"role": "user", "content": prompt}
            ])
        
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating match assessment: {str(e)}")
            return "Impossible de générer une évaluation pour cette correspondance."

    def generate_acceptance_email(self, cv_info: Dict, job_requirements: Dict, match_details: Dict) -> str:
        candidate_name = cv_info['personal_info'].get('name', 'Candidat')
        job_title = job_requirements.get('job_title', 'le poste')
    
        email_content = f"""
        Objet: Candidature retenue pour {job_title}

        Bonjour {candidate_name},

        Nous avons le plaisir de vous informer que votre candidature pour le poste de {job_title} a retenu notre attention.

        Votre profil correspond particulièrement bien à nos attentes, notamment en ce qui concerne :
        """
    
        if match_details['strengths']:
            for strength in match_details['strengths'][:3]:
                email_content += f"- {strength}\n"
    
        email_content += f"""
        Votre score de correspondance global pour ce poste est de {match_details['overall_match']}%, ce qui est très satisfaisant.

        Nous souhaiterions poursuivre le processus de recrutement avec vous lors d'un entretien.
    
        Nous vous contacterons prochainement pour fixer une date qui vous convient.

        Dans l'attente de votre réponse, nous vous prions d'agréer nos salutations distinguées.

        L'équipe de recrutement
        """
    
        return email_content

    def generate_interview_email(self, cv_info: Dict, job_requirements: Dict, interview_date: str) -> str:
        candidate_name = cv_info['personal_info'].get('name', 'Candidat')
        job_title = job_requirements.get('job_title', 'le poste')
    
        meeting_link = "https://meet.google.com/abc-defg-hij"
    
        email_content = f"""
        Objet: Invitation à un entretien pour {job_title}

        Bonjour {candidate_name},

        Suite à notre précédent email concernant votre candidature pour le poste de {job_title},
        nous avons le plaisir de vous inviter à un entretien qui se déroulera le {interview_date}.

        L'entretien aura lieu en visioconférence via Google Meet. Voici le lien pour rejoindre la réunion :
        {meeting_link}

        L'entretien durera environ 45 minutes et se déroulera comme suit :
        - Présentation de l'entreprise et du poste (10 minutes)
        - Discussion sur votre parcours et vos compétences (20 minutes)
        - Questions-réponses (15 minutes)

        Si cette date ou cet horaire ne vous convient pas, merci de nous le faire savoir rapidement
        afin que nous puissions trouver un autre créneau.

         Nous nous réjouissons de vous rencontrer.

        Cordialement,
        L'équipe de recrutement
        """
    
        return email_content

    def generate_rejection_email(self, cv_info: Dict, job_requirements: Dict, match_details: Dict) -> str:
        candidate_name = cv_info['personal_info'].get('name', 'Candidat')
        job_title = job_requirements.get('job_title', 'le poste')
    
        email_content = f"""
    Objet: Réponse concernant votre candidature pour {job_title}

    Bonjour {candidate_name},

    Nous vous remercions de l'intérêt que vous avez porté à notre entreprise et pour votre candidature au poste de {job_title}.

    Après une analyse approfondie de votre profil, nous regrettons de vous informer que nous ne pourrons pas donner suite 
    à votre candidature. Votre score de correspondance global pour ce poste est de {match_details['overall_match']}%.

    Bien que votre profil présente plusieurs qualités notables, telles que :
    """
    
        if match_details['strengths']:
            for strength in match_details['strengths'][:2]:
                email_content += f"- {strength}\n"
        else:
            email_content += "- Votre formation et parcours académique\n"
    
        email_content += """
        Nous recherchons un profil qui corresponde davantage aux exigences spécifiques du poste, notamment concernant :
        """
    
        if match_details['missing_skills']:
            for skill in match_details['missing_skills'][:3]:
                email_content += f"- {skill}\n"
    
        email_content += "\nPour améliorer votre candidature pour ce type de poste, voici quelques recommandations :\n"
    
        if match_details['recommendations']:
            for rec in match_details['recommendations'][:3]:
                email_content += f"- {rec}\n"
        else:
            email_content += """
            - Acquérir une expérience plus spécifique dans les domaines clés mentionnés
            - Suivre des formations complémentaires pour renforcer vos compétences techniques
            - Mettre en avant plus clairement vos réalisations concrètes
            """
    
            email_content += """
        Nous vous souhaitons beaucoup de succès dans votre recherche d'emploi et votre parcours professionnel.

        Cordialement,
        L'équipe de recrutement
        """
    
        return email_content

    def send_email(self, recipient_email: str, subject: str, body: str) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config["sender_email"]
            msg['To'] = recipient_email
            msg['Subject'] = subject
        
            msg.attach(MIMEText(body, 'plain'))
        
            server = smtplib.SMTP(self.smtp_config["smtp_server"], self.smtp_config["smtp_port"])
            server.starttls()
            server.login(self.smtp_config["sender_email"], self.smtp_config["sender_password"])
            text = msg.as_string()
            server.sendmail(self.smtp_config["sender_email"], recipient_email, text)
            server.quit()
        
            logger.info(f"Email sent successfully to {recipient_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False

    def save_xai_report(self, cv_info: Dict, job_requirements: Dict, match_details: Dict, output_path: str) -> bool:
        """
        Sauvegarde le rapport XAI explicatif en HTML.
        
        Args:
            cv_info: Informations du CV
            job_requirements: Exigences du poste
            match_details: Détails de la correspondance
            output_path: Chemin de sortie pour le rapport HTML
            
        Returns:
            Booléen indiquant si la sauvegarde a réussi
        """
        try:
            # Générer le rapport HTML
            xai_report = self.xai.generate_html_report(cv_info, job_requirements, match_details, 
                                                     match_details.get("xai_visuals", {}))
            
            # Sauvegarder le rapport
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xai_report)
                
            logger.info(f"XAI report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving XAI report: {str(e)}")
            return False


def main():
    print("📄 CV Parser & Matcher avec XAI - Analyser votre CV et le comparer aux offres d'emploi")
    print("=" * 80)
    
    smtp_config = {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "gastonishere1000@gmail.com",
        "sender_password": "ijif bzrq gyom mqbl",
        "google_credentials_file": "credentials.json"
    }
    
    try:
        parser = CVParser(smtp_config=smtp_config)
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation de l'application: {str(e)}")
        return
    
    try:
        pdf_path = input("📂 Entrez le chemin vers votre CV (PDF): ")
        
        if not pdf_path:
            print("❌ Le chemin vers le CV est requis.")
            return
            
        print("\n⏳ Analyse du CV en cours... Cela peut prendre quelques instants.")
        cv_info = parser.parse_cv(pdf_path)
        
        print("\n" + "=" * 80)
        print("📊 RÉSULTATS DE L'ANALYSE DU CV 📊")
        print("=" * 80)
        
        print(f"\n👤 INFORMATIONS PERSONNELLES")
        print(f"Nom: {cv_info['personal_info'].get('name', 'Non détecté')}")
        print(f"Email: {cv_info['personal_info'].get('email', 'Non détecté')}")
        print(f"Téléphone: {cv_info['personal_info'].get('phone', 'Non détecté')}")
        print(f"Adresse: {cv_info['personal_info'].get('address', 'Non détectée')}")
        print(f"LinkedIn: {cv_info['personal_info'].get('linkedin', 'Non détecté')}")
        print(f"GitHub: {cv_info['personal_info'].get('github', 'Non détecté')}")
        print(f"Site web: {cv_info['personal_info'].get('website', 'Non détecté')}")
        
        print(f"\n🎓 FORMATION")
        if cv_info['education']:
            for edu in cv_info['education']:
                print(f"• {edu.get('degree', 'Diplôme')} - {edu.get('institution', 'Institution')}")
                print(f"  {edu.get('period', 'Période non spécifiée')} | {edu.get('field', 'Domaine non spécifié')}")
                if edu.get('location'):
                    print(f"  Lieu: {edu.get('location')}")
                if edu.get('grade'):
                    print(f"  Mention: {edu.get('grade')}")
        else:
            print("Aucune information de formation détectée")
        
        print(f"\n💼 EXPÉRIENCE PROFESSIONNELLE")
        if cv_info['experience']:
            for exp in cv_info['experience']:
                print(f"• {exp.get('title', 'Poste')} - {exp.get('company', 'Entreprise')}")
                print(f"  {exp.get('period', 'Période non spécifiée')}")
                if exp.get('location'):
                    print(f"  Lieu: {exp.get('location')}")
                if exp.get('contract_type'):
                    print(f"  Type: {exp.get('contract_type')}")
                resp = exp.get('responsibilities', [])
                if resp:
                    print("  Responsabilités principales:")
                    for r in resp[:5]:
                        print(f"  - {r}")
        else:
            print("Aucune information d'expérience détectée")
            
        print(f"\n🏆 CERTIFICATIONS")
        if cv_info['certifications']:
            for cert in cv_info['certifications']:
                print(f"• {cert.get('name', 'Certification')} - {cert.get('issuer', 'Émetteur')}")
                if cert.get('date'):
                    print(f"  Date: {cert.get('date')}")
                if cert.get('field'):
                    print(f"  Domaine: {cert.get('field')}")
                if cert.get('id'):
                    print(f"  ID: {cert.get('id')}")
                print("")
        else:
            print("Aucune certification détectée")
            
        print(f"\n🚀 PROJETS")
        if cv_info['projects']:
            for proj in cv_info['projects']:
                print(f"• {proj.get('title', 'Projet')}")
                if proj.get('period'):
                    print(f"  Période: {proj.get('period')}")
                if proj.get('description'):
                    print(f"  {proj.get('description')[:100]}...")
                if proj.get('technologies'):
                    tech_list = proj.get('technologies', [])
                    tech_strings = []
                    for tech in tech_list[:5]:
                        if isinstance(tech, str):
                            tech_strings.append(tech)
                        elif isinstance(tech, dict):
                            tech_strings.append(str(tech))
                    print(f"  Technologies: {', '.join(tech_strings)}")
                if proj.get('link'):
                    print(f"  Lien: {proj.get('link')}")
                print("")
        else:
            print("Aucun projet détecté")

        print(f"\n🔧 COMPÉTENCES")
        skills = cv_info['skills']

        print("Compétences techniques:")
        if skills.get('technical_skills'):
            tech_skills = skills.get('technical_skills', [])
            if tech_skills:
                print(", ".join(tech_skills))
        else:
            print("Aucune compétence technique détectée")

        print("\nLangues:")
        if skills.get('languages'):
            for lang in skills.get('languages', []):
                print(f"• {lang}")
        else:
            print("Aucune langue détectée")

        print("\nSoft skills:")
        if skills.get('soft_skills'):
            for skill in skills.get('soft_skills', []):
                print(f"• {skill}")
        else:
            print("Aucun soft skill détecté")

        print(f"\n📝 RÉSUMÉ PROFESSIONNEL")
        print("-" * 80)
        print(cv_info['summary'])
        print("-" * 80)

        match_result = None
        job_description = ""
        job_requirements = {}
        
        matching_option = input("\n🔍 Souhaitez-vous comparer ce CV avec une offre d'emploi? (o/n): ")
        
        if matching_option.lower() in ['o', 'oui', 'y', 'yes']:
            print("\n📋 Entrez la description du poste (terminez par une ligne vide):")
            job_lines = []
            while True:
                line = input()
                if not line:
                    break
                job_lines.append(line)
            
            job_description = "\n".join(job_lines)
            
            if not job_description.strip():
                print("❌ Description du poste vide.")
            else:
                print("\n⏳ Comparaison en cours avec analyse explicable... Cela peut prendre quelques instants.")
                match_result = parser.match_cv_with_job(cv_info, job_description)
                job_requirements = match_result['job_requirements']
                
                print("\n" + "=" * 80)
                print("🎯 RÉSULTATS DE LA CORRESPONDANCE EXPLIQUÉE 🎯")
                print("=" * 80)
                
                print(f"\n📊 SCORES DE CORRESPONDANCE")
                print(f"Global: {match_result['overall_match']}%")
                print(f"Compétences techniques: {match_result['technical_skills_match']}%")
                print(f"Formation: {match_result['education_match']}%")
                print(f"Expérience: {match_result['experience_match']}%")
                print(f"Langues: {match_result['languages_match']}%")
                print(f"Soft skills: {match_result['soft_skills_match']}%")
                print(f"Correspondance du domaine: {match_result.get('domain_match', 'Non évalué')}%")
                
                print(f"\n❓ COMPÉTENCES MANQUANTES")
                if match_result['missing_skills']:
                    for skill in match_result['missing_skills']:
                        print(f"• {skill}")
                else:
                    print("Aucune compétence clé manquante détectée")
                
                print(f"\n💪 POINTS FORTS")
                if match_result['strengths']:
                    for strength in match_result['strengths']:
                        print(f"• {strength}")
                else:
                    print("Aucun point fort spécifique identifié")
                
                print(f"\n📌 RECOMMANDATIONS")
                if match_result['recommendations']:
                    for rec in match_result['recommendations']:
                        print(f"• {rec}")
                else:
                    print("Aucune recommandation spécifique")
                
                print(f"\n📝 ÉVALUATION FINALE")
                print("-" * 80)
                print(match_result['final_assessment'])
                print("-" * 80)
                
                # Générer et sauvegarder le rapport XAI
                xai_report_option = input("\n🔍 Souhaitez-vous générer un rapport d'explication détaillé (XAI)? (o/n): ")
                if xai_report_option.lower() in ['o', 'oui', 'y', 'yes']:
                    xai_output_path = f"{os.path.splitext(pdf_path)[0]}_rapport_xai.html"
                    if parser.save_xai_report(cv_info, job_requirements, match_result, xai_output_path):
                        print(f"✅ Rapport d'explication XAI sauvegardé dans: {xai_output_path}")
                        print("   Ce rapport contient des visualisations interactives expliquant la correspondance.")
                    else:
                        print("❌ Erreur lors de la sauvegarde du rapport XAI.")
                
                email_option = input("\n✉️ Souhaitez-vous générer un email pour le candidat? (o/n): ")
                if email_option.lower() in ['o', 'oui', 'y', 'yes']:
                    if match_result['overall_match'] >= 70:
                        acceptance_option = input("\n🎯 Le score de correspondance est bon. Accepter la candidature? (o/n): ")
                        if acceptance_option.lower() in ['o', 'oui', 'y', 'yes']:
                            acceptance_email = parser.generate_acceptance_email(cv_info, job_requirements, match_result)
                            print("\n" + "=" * 80)
                            print("📧 EMAIL D'ACCEPTATION")
                            print("=" * 80)
                            print(acceptance_email)
                            
                            schedule_interview = input("\n📅 Souhaitez-vous programmer un entretien? (o/n): ")
                            if schedule_interview.lower() in ['o', 'oui', 'y', 'yes']:
                                interview_date = input("Entrez la date et l'heure de l'entretien (ex: 25 avril 2025 à 14h30): ")
                                interview_email = parser.generate_interview_email(cv_info, job_requirements, interview_date)
                                print("\n" + "=" * 80)
                                print("📧 EMAIL D'INVITATION À L'ENTRETIEN")
                                print("=" * 80)
                                print(interview_email)
                                
                                send_option = input("\n📤 Envoyer les emails au candidat? (o/n): ")
                                if send_option.lower() in ['o', 'oui', 'y', 'yes']:
                                    candidate_email = cv_info['personal_info'].get('email')
                                    if not candidate_email:
                                        candidate_email = input("Email du candidat non détecté. Veuillez l'entrer: ")
                                    
                                    if candidate_email:
                                        subject1 = f"Candidature retenue pour {job_requirements.get('job_title', 'le poste')}"
                                        sent1 = parser.send_email(candidate_email, subject1, acceptance_email)
                                        
                                        subject2 = f"Invitation à un entretien pour {job_requirements.get('job_title', 'le poste')}"
                                        sent2 = parser.send_email(candidate_email, subject2, interview_email)
                                        
                                        if sent1 and sent2:
                                            print("✅ Les emails ont été envoyés avec succès!")
                                        else:
                                            print("❌ Erreur lors de l'envoi des emails.")
                                    else:
                                        print("❌ Aucune adresse email fournie.")
                        else:
                            rejection = input("\n❌ Souhaitez-vous envoyer un email de refus? (o/n): ")
                            if rejection.lower() in ['o', 'oui', 'y', 'yes']:
                                rejection_email = parser.generate_rejection_email(cv_info, job_requirements, match_result)
                                print("\n" + "=" * 80)
                                print("📧 EMAIL DE REFUS")
                                print("=" * 80)
                                print(rejection_email)
                                
                                send_option = input("\n📤 Envoyer l'email au candidat? (o/n): ")
                                if send_option.lower() in ['o', 'oui', 'y', 'yes']:
                                    candidate_email = cv_info['personal_info'].get('email')
                                    if not candidate_email:
                                        candidate_email = input("Email du candidat non détecté. Veuillez l'entrer: ")
                                    
                                    if candidate_email:
                                        subject = f"Réponse concernant votre candidature pour {job_requirements.get('job_title', 'le poste')}"
                                        sent = parser.send_email(candidate_email, subject, rejection_email)
                                        
                                        if sent:
                                            print("✅ L'email a été envoyé avec succès!")
                                        else:
                                            print("❌ Erreur lors de l'envoi de l'email.")
                                    else:
                                        print("❌ Aucune adresse email fournie.")
                    else:
                        print("\n⚠️ Le score de correspondance est faible (inférieur à 70%).")
                        rejection_option = input("Souhaitez-vous générer un email de refus? (o/n): ")
                        if rejection_option.lower() in ['o', 'oui', 'y', 'yes']:
                            rejection_email = parser.generate_rejection_email(cv_info, job_requirements, match_result)
                            print("\n" + "=" * 80)
                            print("📧 EMAIL DE REFUS")
                            print("=" * 80)
                            print(rejection_email)
                            
                            send_option = input("\n📤 Envoyer l'email au candidat? (o/n): ")
                            if send_option.lower() in ['o', 'oui', 'y', 'yes']:
                                candidate_email = cv_info['personal_info'].get('email')
                                if not candidate_email:
                                    candidate_email = input("Email du candidat non détecté. Veuillez l'entrer: ")
                                
                                if candidate_email:
                                    subject = f"Réponse concernant votre candidature pour {job_requirements.get('job_title', 'le poste')}"
                                    sent = parser.send_email(candidate_email, subject, rejection_email)
                                    
                                    if sent:
                                        print("✅ L'email a été envoyé avec succès!")
                                    else:
                                        print("❌ Erreur lors de l'envoi de l'email.")
                                else:
                                    print("❌ Aucune adresse email fournie.")
                
                match_save_option = input("\n💾 Souhaitez-vous sauvegarder les résultats du matching? (o/n): ")
                if match_save_option.lower() in ['o', 'oui', 'y', 'yes']:
                    match_output_path = f"{os.path.splitext(pdf_path)[0]}_matching.json"
                    try:
                        # Enlever les visuels XAI avant sauvegarde JSON (ils sont encodés en base64)
                        match_result_to_save = match_result.copy()
                        if "xai_visuals" in match_result_to_save:
                            del match_result_to_save["xai_visuals"]
                            
                        with open(match_output_path, 'w', encoding='utf-8') as f:
                            json.dump(match_result_to_save, f, ensure_ascii=False, indent=4)
                        print(f"✅ Résultats du matching sauvegardés dans: {match_output_path}")
                    except Exception as e:
                        print(f"❌ Erreur lors de la sauvegarde: {str(e)}")

        save_option = input("\n💾 Souhaitez-vous sauvegarder l'analyse du CV? (o/n): ")
        if save_option.lower() in ['o', 'oui', 'y', 'yes']:
            output_path = f"{os.path.splitext(pdf_path)[0]}_analyse.json"
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(cv_info, f, ensure_ascii=False, indent=4)
                print(f"✅ Résultats sauvegardés dans: {output_path}")
            except Exception as e:
                print(f"❌ Erreur lors de la sauvegarde: {str(e)}")

        print("\n" + "=" * 80)
        print("Merci d'avoir utilisé notre analyseur et comparateur de CV avec explication XAI!")
        print("=" * 80)
            
    except Exception as e:
        print(f"❌ Une erreur est survenue: {str(e)}")
        logger.error(f"Error in main function: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()