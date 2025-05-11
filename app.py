import os
import json
import secrets
from datetime import datetime, timedelta  # Correct import - using only this one
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, abort, jsonify
import logging
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from avatar3 import AITechnicalInterviewer
import os
import json
import base64
import cv2
import numpy as np
import mediapipe as mp
from dotenv import load_dotenv
import threading
import queue
import time
from phone_detector import load_phone_model, detect_phone_in_frame, save_fraud_detection_image
from emotion_detection import load_emotion_model, detect_emotion_in_frame, save_emotion_detection_image
# Removed the conflicting import: import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, abort, jsonify, current_app

# Importer le parser de CV
from main import CVParser
phone_model = load_phone_model()
emotion_model = load_emotion_model()
# Charger les variables d'environnement
load_dotenv()
frame_counter = 0
# Importer les fonctions du chatbot
from chatbot_api2 import (
    ensure_directories, 
    process_query, 
    process_audio_query,  # This should now work once you add the function
    upload_document,
    list_agents,
    query_specific_agent,
    get_documents_and_faiss_index,
    get_embedder
)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
socketio = SocketIO(app, cors_allowed_origins="*")


# Configuration de l'application
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
JOB_OFFERS_FOLDER = 'job_offers'
APPLICATIONS_FOLDER = 'applications'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JOB_OFFERS_FOLDER'] = JOB_OFFERS_FOLDER
app.config['APPLICATIONS_FOLDER'] = APPLICATIONS_FOLDER
app.config['ADMIN_USERNAME'] = 'admin'
app.config['ADMIN_PASSWORD'] = 'admin123'  # À changer en production!
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16 Mo pour les uploads

# Configuration SMTP pour les emails
SMTP_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "gastonishere1000@gmail.com",
    "sender_password": "ijif bzrq gyom mqbl",  
    "google_credentials_file": "credentials.json"
}

# Créer les dossiers nécessaires s'ils n'existent pas
for folder in [UPLOAD_FOLDER, JOB_OFFERS_FOLDER, APPLICATIONS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Catégories d'emploi
CATEGORIES = {
    'stage_ete': 'Stage d\'été',
    'stage_pfe': 'Stage PFE',
    'embauche': 'Embauche',
    'alternance': 'Alternance'
}

# Initialisation du parser de CV
cv_parser = CVParser(smtp_config=SMTP_CONFIG)

def allowed_file(filename):
    """Vérifie si le fichier a une extension autorisée."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_job_offers(category):
    """Récupère les offres d'emploi d'une catégorie."""
    category_folder = os.path.join(app.config['JOB_OFFERS_FOLDER'], category)
    os.makedirs(category_folder, exist_ok=True)
    
    job_offers = []
    for filename in os.listdir(category_folder):
        if filename.endswith('.json'):
            with open(os.path.join(category_folder, filename), 'r', encoding='utf-8') as f:
                job_offer = json.load(f)
                job_offer['id'] = filename.split('.')[0]
                job_offers.append(job_offer)
    
    return job_offers

def get_applications():
    """Récupère toutes les candidatures."""
    applications = []
    for filename in os.listdir(app.config['APPLICATIONS_FOLDER']):
        if filename.endswith('.json'):
            with open(os.path.join(app.config['APPLICATIONS_FOLDER'], filename), 'r', encoding='utf-8') as f:
                try:
                    application = json.load(f)
                    application['id'] = filename.split('.')[0]
                    
                    # Ajouter le score global s'il existe dans match_result
                    if 'match_result' in application and 'overall_match' in application['match_result']:
                        application['score'] = application['match_result']['overall_match']
                    # Si le score est directement stocké dans l'application
                    elif 'score' in application:
                        application['score'] = application['score']
                    else:
                        application['score'] = 0
                        
                    applications.append(application)
                except json.JSONDecodeError:
                    app.logger.error(f"Erreur de décodage JSON pour le fichier {filename}")
                    continue
    
    # Tri des candidatures par score décroissant
    applications.sort(key=lambda x: x['score'], reverse=True)
    
    return applications

def is_admin():
    """Vérifie si l'utilisateur courant est admin."""
    return 'admin' in session and session['admin']

def generate_access_code():
    """Génère un code d'accès aléatoire pour l'entretien virtuel."""
    # Format: 6 caractères alphanumériques en majuscules
    return ''.join(secrets.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(6))

@app.route('/')
def index():
    """Page d'accueil."""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Page de connexion admin."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == app.config['ADMIN_USERNAME'] and password == app.config['ADMIN_PASSWORD']:
            session['admin'] = True
            flash('Connexion réussie!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Nom d\'utilisateur ou mot de passe incorrect!', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Déconnexion de l'admin."""
    session.pop('admin', None)
    flash('Vous avez été déconnecté.', 'info')
    return redirect(url_for('index'))

@app.route('/admin')
@app.route('/admin/')
def admin_dashboard():
    """Dashboard de l'admin."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    # Récupérer les paramètres de filtrage et de tri
    filter_status = request.args.get('filter')
    sort_option = request.args.get('sort', 'score')  # Par défaut, tri par score
    
    categories = CATEGORIES
    applications = get_applications()
    
    # Appliquer le filtre par statut si spécifié
    if filter_status:
        applications = [app for app in applications if app.get('status') == filter_status]
    
    # Appliquer le tri
    if sort_option == 'date':
        # Tri par date (plus récent d'abord)
        applications.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    elif sort_option == 'alpha':
        # Tri alphabétique par nom
        applications.sort(key=lambda x: x.get('name', '').lower())
    # Le tri par score est déjà appliqué dans get_applications()
    
    return render_template('admin_dashboard.html', categories=categories, applications=applications)

@app.route('/admin/category/<category>')
@app.route('/admin/category/<category>/')
def admin_category(category):
    """Affiche les offres d'emploi d'une catégorie pour l'admin."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    if category not in CATEGORIES:
        flash('Catégorie non valide!', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    job_offers = get_job_offers(category)
    
    return render_template('admin_category.html', 
                          category=category, 
                          category_name=CATEGORIES[category], 
                          job_offers=job_offers)

@app.route('/admin/job/new/<category>', methods=['GET', 'POST'])
@app.route('/admin/job/new/<category>/', methods=['GET', 'POST'])
def admin_new_job(category):
    """Crée une nouvelle offre d'emploi."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    if category not in CATEGORIES:
        flash('Catégorie non valide!', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        job_title = request.form.get('job_title')
        job_description = request.form.get('job_description')
        
        if not job_title or not job_description:
            flash('Tous les champs sont obligatoires!', 'danger')
        else:
            job_id = f"{int(datetime.now().timestamp())}"  # This will now work correctly
            job_data = {
                'title': job_title,
                'description': job_description,
                'category': category,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # This will also work correctly
            }
            
            category_folder = os.path.join(app.config['JOB_OFFERS_FOLDER'], category)
            os.makedirs(category_folder, exist_ok=True)
            
            with open(os.path.join(category_folder, f"{job_id}.json"), 'w', encoding='utf-8') as f:
                json.dump(job_data, f, ensure_ascii=False, indent=4)
            
            flash('Offre d\'emploi créée avec succès!', 'success')
            return redirect(url_for('admin_category', category=category))
    
    return render_template('admin_new_job.html', category=category, category_name=CATEGORIES[category])

@app.route('/admin/job/edit/<category>/<job_id>', methods=['GET', 'POST'])
@app.route('/admin/job/edit/<category>/<job_id>/', methods=['GET', 'POST'])
def admin_edit_job(category, job_id):
    """Modifie une offre d'emploi existante."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], category, f"{job_id}.json")
    
    if not os.path.exists(job_path):
        flash('Offre d\'emploi non trouvée!', 'danger')
        return redirect(url_for('admin_category', category=category))
    
    with open(job_path, 'r', encoding='utf-8') as f:
        job_data = json.load(f)
    
    if request.method == 'POST':
        job_title = request.form.get('job_title')
        job_description = request.form.get('job_description')
        
        if not job_title or not job_description:
            flash('Tous les champs sont obligatoires!', 'danger')
        else:
            job_data['title'] = job_title
            job_data['description'] = job_description
            job_data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(job_path, 'w', encoding='utf-8') as f:
                json.dump(job_data, f, ensure_ascii=False, indent=4)
            
            flash('Offre d\'emploi modifiée avec succès!', 'success')
            return redirect(url_for('admin_category', category=category))
    
    return render_template('admin_edit_job.html', 
                          category=category, 
                          category_name=CATEGORIES[category], 
                          job=job_data)

@app.route('/admin/job/delete/<category>/<job_id>')
@app.route('/admin/job/delete/<category>/<job_id>/')
def admin_delete_job(category, job_id):
    """Supprime une offre d'emploi et les candidatures associées."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], category, f"{job_id}.json")
    
    if os.path.exists(job_path):
        # Vérifier si des candidatures sont liées à cette offre
        linked_applications = []
        for filename in os.listdir(app.config['APPLICATIONS_FOLDER']):
            if filename.endswith('.json'):
                with open(os.path.join(app.config['APPLICATIONS_FOLDER'], filename), 'r', encoding='utf-8') as f:
                    try:
                        application = json.load(f)
                        if application.get('job_id') == job_id and application.get('category') == category:
                            application['id'] = filename.split('.')[0]  # Ajouter l'ID à l'objet application
                            linked_applications.append(application)
                    except json.JSONDecodeError:
                        continue
        
        # Si des candidatures sont liées et qu'aucune confirmation n'a été donnée
        if linked_applications and not request.args.get('confirm'):
            return render_template('admin_confirm_delete_job.html', 
                                  category=category,
                                  category_name=CATEGORIES[category],
                                  job_id=job_id,
                                  linked_applications=linked_applications)
        
        # Si confirmation ou pas de candidatures liées
        # Supprimer l'offre d'emploi
        os.remove(job_path)
        
        # Supprimer toutes les candidatures associées à cette offre
        deleted_count = 0
        for application in linked_applications:
            app_id = application.get('id', '')
            
            if app_id:
                application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{app_id}.json")
                # Supprimer le fichier CV associé s'il existe
                cv_filename = application.get('cv_file')
                if cv_filename:
                    cv_path = os.path.join(app.config['UPLOAD_FOLDER'], cv_filename)
                    if os.path.exists(cv_path):
                        try:
                            os.remove(cv_path)
                        except Exception as e:
                            app.logger.error(f"Erreur lors de la suppression du CV {cv_path}: {str(e)}")
                
                # Supprimer le fichier de candidature
                if os.path.exists(application_path):
                    os.remove(application_path)
                    deleted_count += 1
        
        if deleted_count > 0:
            flash(f'{deleted_count} candidature(s) associée(s) supprimée(s) avec succès!', 'info')
        
        flash('Offre d\'emploi supprimée avec succès!', 'success')
    else:
        flash('Offre d\'emploi non trouvée!', 'danger')
    
    return redirect(url_for('admin_category', category=category))

@app.route('/jobs')
@app.route('/jobs/')
def public_jobs():
    """Affiche toutes les offres d'emploi disponibles."""
    # Récupérer le paramètre de catégorie depuis l'URL si présent
    category_filter = request.args.get('category')
    
    all_jobs = []
    
    for category, category_name in CATEGORIES.items():
        # Si un filtre de catégorie est appliqué, ne récupérer que les offres de cette catégorie
        if category_filter and category != category_filter:
            continue
            
        jobs = get_job_offers(category)
        for job in jobs:
            job['category_name'] = category_name
        all_jobs.extend(jobs)
    
    return render_template('public_jobs.html', jobs=all_jobs, categories=CATEGORIES)


@app.route('/job/<category>/<job_id>')
@app.route('/job/<category>/<job_id>/')
def public_job_detail(category, job_id):
    """Affiche les détails d'une offre d'emploi."""
    job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], category, f"{job_id}.json")
    
    if not os.path.exists(job_path):
        flash('Offre d\'emploi non trouvée!', 'danger')
        return redirect(url_for('public_jobs'))
    
    with open(job_path, 'r', encoding='utf-8') as f:
        job_data = json.load(f)
    
    return render_template('public_job_detail.html', 
                          job=job_data, 
                          job_id=job_id,
                          category=category, 
                          category_name=CATEGORIES[category])

@app.route('/apply/<category>/<job_id>', methods=['GET', 'POST'])
@app.route('/apply/<category>/<job_id>/', methods=['GET', 'POST'])
def apply_job(category, job_id):
    """Permet de postuler à une offre d'emploi."""
    job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], category, f"{job_id}.json")
    
    if not os.path.exists(job_path):
        flash('Offre d\'emploi non trouvée!', 'danger')
        return redirect(url_for('public_jobs'))
    
    with open(job_path, 'r', encoding='utf-8') as f:
        job_data = json.load(f)
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        
        # Vérifier si le candidat a déjà postulé à cette offre
        already_applied = False
        for filename in os.listdir(app.config['APPLICATIONS_FOLDER']):
            if filename.endswith('.json'):
                with open(os.path.join(app.config['APPLICATIONS_FOLDER'], filename), 'r', encoding='utf-8') as f:
                    application = json.load(f)
                    if (application.get('email') == email and 
                        application.get('job_id') == job_id and 
                        application.get('category') == category):
                        already_applied = True
                        break
        
        if already_applied:
            flash('Vous avez déjà postulé à cette offre avec cette adresse email!', 'warning')
            return redirect(url_for('public_job_detail', category=category, job_id=job_id))
        
        # Vérifier si le fichier a été soumis
        if 'cv_file' not in request.files:
            flash('Aucun fichier CV trouvé!', 'danger')
            return redirect(request.url)
        
        file = request.files['cv_file']
        
        # Si l'utilisateur ne sélectionne pas de fichier, le navigateur soumet un fichier vide
        if file.filename == '':
            flash('Aucun fichier sélectionné!', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Sécuriser le nom de fichier et l'enregistrer
            filename = secure_filename(file.filename)
            timestamp = int(datetime.now().timestamp())
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Créer l'entrée de candidature
            application_id = f"{timestamp}"
            application_data = {
                'name': name,
                'email': email,
                'job_id': job_id,
                'job_title': job_data['title'],
                'category': category,
                'cv_file': unique_filename,
                'status': 'pending',
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Enregistrer la candidature
            with open(os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_id}.json"), 'w', encoding='utf-8') as f:
                json.dump(application_data, f, ensure_ascii=False, indent=4)
            
            flash('Votre candidature a été soumise avec succès! Vous recevrez une réponse par email.', 'success')
            return redirect(url_for('application_success'))
        else:
            flash('Type de fichier non autorisé! Seuls les fichiers PDF sont acceptés.', 'danger')
    
    return render_template('apply_job.html', 
                          job=job_data, 
                          job_id=job_id,
                          category=category, 
                          category_name=CATEGORIES[category])

@app.route('/application/success')
@app.route('/application/success/')
def application_success():
    """Page de confirmation après soumission d'une candidature."""
    return render_template('application_success.html')

@app.route('/admin/applications')
@app.route('/admin/applications/')
def admin_applications_list():
    """Liste toutes les candidatures pour l'admin."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    applications = get_applications()
    
    return render_template('admin_applications.html', 
                          applications=applications, 
                          categories=CATEGORIES)

@app.route('/admin/application')
@app.route('/admin/application/')
def admin_application_redirect():
    """Redirige vers la liste des candidatures."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    flash('Veuillez sélectionner une candidature spécifique à consulter.', 'info')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/application/<application_id>', methods=['GET', 'POST'])
@app.route('/admin/application/<application_id>/', methods=['GET', 'POST'])
def admin_application_detail(application_id):
    """Affiche les détails d'une candidature pour l'admin."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_id}.json")
    
    if not os.path.exists(application_path):
        flash('Candidature non trouvée!', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    # Gérer la suppression si le paramètre action=delete est présent
    if request.args.get('action') == 'delete':
        # Si une confirmation est demandée mais pas encore donnée
        if not request.args.get('confirm'):
            with open(application_path, 'r', encoding='utf-8') as f:
                application_data = json.load(f)
                # Assurons-nous que l'ID est inclus dans les données de candidature
                application_data['id'] = application_id
            # MODIFICATION: Utiliser un template qui n'étend pas admin_base.html
            return render_template('delete_application_confirm.html', application=application_data)
        
        # Si la confirmation est donnée, supprimer le fichier
        with open(application_path, 'r', encoding='utf-8') as f:
            application_data = json.load(f)
        
        # Supprimer le fichier CV associé s'il existe
        cv_filename = application_data.get('cv_file')
        if cv_filename:
            cv_path = os.path.join(app.config['UPLOAD_FOLDER'], cv_filename)
            if os.path.exists(cv_path):
                try:
                    os.remove(cv_path)
                except Exception as e:
                    app.logger.error(f"Erreur lors de la suppression du CV {cv_path}: {str(e)}")
        
        # Supprimer le fichier de candidature
        os.remove(application_path)
        flash('Candidature supprimée avec succès!', 'success')
        return redirect(url_for('admin_dashboard'))
    
    with open(application_path, 'r', encoding='utf-8') as f:
        application_data = json.load(f)
    
    # Gérer les soumissions POST (si vous avez un formulaire sur cette page)
    if request.method == 'POST':
        # Traiter les données du formulaire ici
        action = request.form.get('action')
        if action == 'update_notes':
            notes = request.form.get('notes')
            application_data['admin_notes'] = notes
            with open(application_path, 'w', encoding='utf-8') as f:
                json.dump(application_data, f, ensure_ascii=False, indent=4)
            flash('Notes mises à jour avec succès!', 'success')
    
    # Récupérer le chemin du fichier CV
    cv_path = os.path.join(app.config['UPLOAD_FOLDER'], application_data['cv_file'])
    
    cv_info = {}
    match_result = {}
    
    # Si le fichier CV existe, l'analyser
    if os.path.exists(cv_path):
        try:
            # Analyser le CV
            cv_info = cv_parser.parse_cv(cv_path)
            
            # Récupérer l'offre d'emploi associée
            job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], 
                                   application_data['category'], 
                                   f"{application_data['job_id']}.json")
            
            if os.path.exists(job_path):
                with open(job_path, 'r', encoding='utf-8') as f:
                    job_data = json.load(f)
                
                # Faire la correspondance entre le CV et l'offre d'emploi
                match_result = cv_parser.match_cv_with_job(cv_info, job_data['description'])
                
                # Sauvegarder les résultats de correspondance
                application_data['cv_info'] = cv_info
                application_data['match_result'] = match_result
                
                # Sauvegarder le score global séparément pour le conserver même en cas de rejet
                if 'overall_match' in match_result:
                    application_data['score'] = match_result['overall_match']
                
                with open(application_path, 'w', encoding='utf-8') as f:
                    json.dump(application_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            flash(f'Erreur lors de l\'analyse du CV: {str(e)}', 'danger')
    else:
        flash('Fichier CV non trouvé!', 'danger')
    
    # Si l'analyse du CV a déjà été faite
    if 'cv_info' in application_data and 'match_result' in application_data:
        cv_info = application_data['cv_info']
        match_result = application_data['match_result']
    
    return render_template('admin_application_detail.html', 
                          application=application_data, 
                          cv_info=cv_info, 
                          match_result=match_result,
                          categories=CATEGORIES)


@app.route('/admin/application/accept')
@app.route('/admin/application/accept/')
def admin_accept_application_redirect():
    """Redirige vers la liste des candidatures."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    flash('Veuillez sélectionner une candidature spécifique à accepter.', 'info')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/application/accept/<application_id>', methods=['GET', 'POST'])
@app.route('/admin/application/accept/<application_id>/', methods=['GET', 'POST'])
def admin_accept_application(application_id):
    """Accepte une candidature et envoie un email d'acceptation, suivi d'un email de programmation d'entretien."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_id}.json")
    
    if not os.path.exists(application_path):
        flash('Candidature non trouvée!', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    with open(application_path, 'r', encoding='utf-8') as f:
        application_data = json.load(f)
    
    if request.method == 'POST':
        # Récupérer les données du formulaire pour l'acceptation
        acceptance_message = request.form.get('acceptance_message')
        
        # Récupérer les données du formulaire pour l'entretien
        interview_date = request.form.get('interview_date')
        google_meet_link = request.form.get('google_meet_link')
        interview_message = request.form.get('interview_message')
        access_code = request.form.get('access_code')
        
        # Générer un code d'accès si aucun n'est fourni
        if not access_code:
            access_code = generate_access_code()
        
        # Vérifier que la date d'entretien et le lien Meet sont fournis
        if not interview_date or not google_meet_link:
            flash('La date d\'entretien et le lien Google Meet sont obligatoires!', 'danger')
            return render_template('admin_accept_application.html', application=application_data)
        
        try:
            # Récupérer les informations nécessaires
            job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], 
                                   application_data['category'], 
                                   f"{application_data['job_id']}.json")
            
            with open(job_path, 'r', encoding='utf-8') as f:
                job_data = json.load(f)
            
            # Préserver le score existant
            score = application_data.get('score')
            if not score and 'match_result' in application_data and 'overall_match' in application_data['match_result']:
                score = application_data['match_result']['overall_match']
            
            # Convertir le format de date si nécessaire pour un affichage plus lisible
            formatted_date = interview_date
            try:
                # Si la date est au format datetime-local (YYYY-MM-DDThh:mm)
                if 'T' in interview_date:
                    datetime_obj = datetime.strptime(interview_date, '%Y-%m-%dT%H:%M')
                    formatted_date = datetime_obj.strftime('%A %d %B %Y à %H:%M')
            except:
                pass
            
            # 1. PREMIER EMAIL : Acceptation de la candidature
            if not acceptance_message:
                # Utiliser l'email généré par le système
                cv_info = application_data.get('cv_info', {})
                match_result = application_data.get('match_result', {})
                
                if cv_info and match_result:
                    acceptance_email = cv_parser.generate_acceptance_email(cv_info, match_result.get('job_requirements', {}), match_result)
                else:
                    acceptance_email = f"""
                    Objet: Candidature retenue pour {job_data['title']}

                    Bonjour {application_data['name']},

                    Nous avons le plaisir de vous informer que votre candidature pour le poste de {job_data['title']} a retenu notre attention.

                    Votre profil correspond particulièrement bien à nos attentes.

                    Nous vous invitons à consulter notre prochain email qui contient tous les détails concernant l'entretien que nous souhaitons programmer avec vous.

                    Dans l'attente de cet échange, nous vous prions d'agréer nos salutations distinguées.

                    L'équipe de recrutement
                    """
            else:
                acceptance_email = acceptance_message
            
            # 2. DEUXIÈME EMAIL : Programmation de l'entretien
            if not interview_message:                
                interview_email = f"""
                Objet: Invitation à un entretien pour {job_data['title']}

                Bonjour {application_data['name']},

                Suite à notre précédent email concernant votre candidature pour le poste de {job_data['title']},
                nous avons le plaisir de vous inviter à un entretien qui se déroulera le {formatted_date}.

                L'entretien aura lieu en visioconférence via Google Meet. Voici le lien pour rejoindre la réunion :
                {google_meet_link}

                Pour accéder à notre plateforme d'entretien virtuel, veuillez utiliser le code d'accès unique suivant :
                CODE D'ACCÈS : {access_code}

                URL de l'entretien virtuel : {request.host_url}virtual_interview

                L'entretien durera environ 45 minutes et se déroulera comme suit :
                - Présentation de l'entreprise et du poste (10 minutes)
                - Discussion sur votre parcours et vos compétences (20 minutes)
                - Questions-réponses (15 minutes)

                Veuillez confirmer votre disponibilité pour cet entretien en répondant à cet e-mail.

                Si cette date ou cet horaire ne vous convient pas, merci de nous le faire savoir rapidement
                afin que nous puissions trouver un autre créneau.

                Nous nous réjouissons de vous rencontrer et d'en apprendre davantage sur votre profil.

                Cordialement,
                L'équipe de recrutement
                """
            else:
                interview_email = interview_message
            
            # Envoyer le premier email (acceptation)
            cv_parser.send_email(
                application_data['email'],
                f"Candidature retenue pour {job_data['title']}",
                acceptance_email
            )
            
            # Envoyer le deuxième email (entretien)
            cv_parser.send_email(
                application_data['email'],
                f"Invitation à un entretien pour {job_data['title']}",
                interview_email
            )
            
            # Mettre à jour le statut de la candidature (directement en mode entretien)
            application_data['status'] = 'interview'
            application_data['response_email'] = acceptance_email
            application_data['response_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            application_data['interview_date'] = formatted_date
            application_data['interview_link'] = google_meet_link
            application_data['interview_email'] = interview_email
            application_data['interview_scheduled_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            application_data['access_code'] = access_code
            
            # S'assurer que le score est préservé
            if score:
                application_data['score'] = score
            
            with open(application_path, 'w', encoding='utf-8') as f:
                json.dump(application_data, f, ensure_ascii=False, indent=4)
            
            flash('Les emails d\'acceptation et d\'invitation à l\'entretien ont été envoyés avec succès!', 'success')
            return redirect(url_for('admin_dashboard'))
        except Exception as e:
            flash(f'Erreur lors de l\'envoi des emails: {str(e)}', 'danger')
    
    return render_template('admin_accept_application.html', application=application_data)

@app.route('/admin/application/reject')
@app.route('/admin/application/reject/')
def admin_reject_application_redirect():
    """Redirige vers la liste des candidatures."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    flash('Veuillez sélectionner une candidature spécifique à rejeter.', 'info')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/application/reject/<application_id>', methods=['GET', 'POST'])
@app.route('/admin/application/reject/<application_id>/', methods=['GET', 'POST'])
def admin_reject_application(application_id):
    """Rejette une candidature et envoie un email."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_id}.json")
    
    if not os.path.exists(application_path):
        flash('Candidature non trouvée!', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    with open(application_path, 'r', encoding='utf-8') as f:
        application_data = json.load(f)
    
    if request.method == 'POST':
        message = request.form.get('message')
        
        try:
            # Récupérer les informations nécessaires
            job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], 
                                   application_data['category'], 
                                   f"{application_data['job_id']}.json")
            
            with open(job_path, 'r', encoding='utf-8') as f:
                job_data = json.load(f)
            
            # Préserver le score existant
            score = application_data.get('score')
            if not score and 'match_result' in application_data and 'overall_match' in application_data['match_result']:
                score = application_data['match_result']['overall_match']
                
            # Générer l'email de rejet
            if not message:
                # Utiliser l'email généré par le système
                cv_info = application_data.get('cv_info', {})
                match_result = application_data.get('match_result', {})
                
                if cv_info and match_result:
                    email_content = cv_parser.generate_rejection_email(cv_info, match_result.get('job_requirements', {}), match_result)
                else:
                    email_content = f"""
                    Objet: Réponse concernant votre candidature pour {job_data['title']}

                    Bonjour {application_data['name']},

                    Nous vous remercions de l'intérêt que vous avez porté à notre entreprise et pour votre candidature au poste de {job_data['title']}.

                    Après une analyse approfondie de votre profil, nous regrettons de vous informer que nous ne pourrons pas donner suite 
                    à votre candidature.

                    Nous vous souhaitons beaucoup de succès dans votre recherche d'emploi et votre parcours professionnel.

                    Cordialement,
                    L'équipe de recrutement
                    """
            else:
                email_content = message
            
            # Envoyer l'email
            cv_parser.send_email(
                application_data['email'],
                f"Réponse concernant votre candidature pour {job_data['title']}",
                email_content
            )
            
            # Mettre à jour le statut de la candidature
            application_data['status'] = 'rejected'
            application_data['response_email'] = email_content
            application_data['response_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # S'assurer que le score est préservé même après le rejet
            if score:
                application_data['score'] = score
            
            with open(application_path, 'w', encoding='utf-8') as f:
                json.dump(application_data, f, ensure_ascii=False, indent=4)
            
            flash('Email de rejet envoyé avec succès!', 'success')
            return redirect(url_for('admin_dashboard'))
        except Exception as e:
            flash(f'Erreur lors de l\'envoi de l\'email: {str(e)}', 'danger')
    
    return render_template('admin_reject_application.html', application=application_data)

@app.route('/uploads/<filename>')
def download_file(filename):
    """Télécharge un fichier."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/admin/schedule_interview')
@app.route('/admin/schedule_interview/')
def admin_schedule_interview_redirect():
    """Redirige vers la liste des candidatures."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    # Au lieu de simplement rediriger, montrer une liste de candidats 
    # pour lesquels on peut programmer un entretien
    applications = get_applications()
    # Filtrer pour ne montrer que les candidatures en attente ou acceptées
    eligible_applications = [app for app in applications if app.get('status') in ['pending', 'accepted']]
    
    return render_template('admin_select_candidate.html', 
                          applications=eligible_applications, 
                          categories=CATEGORIES)

@app.route('/admin/schedule_interview/<application_id>', methods=['GET', 'POST'])
@app.route('/admin/schedule_interview/<application_id>/', methods=['GET', 'POST'])
def admin_schedule_interview(application_id):
    """Planifie un entretien et envoie un email."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_id}.json")
    
    if not os.path.exists(application_path):
        flash('Candidature non trouvée!', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    with open(application_path, 'r', encoding='utf-8') as f:
        application_data = json.load(f)
    
    if request.method == 'POST':
        interview_date = request.form.get('interview_date')
        interview_type = request.form.get('interview_type', 'video')
        interview_location = request.form.get('interview_location', '')
        message = request.form.get('message')
        access_code = request.form.get('access_code')
        
        # Générer un code d'accès si aucun n'est fourni
        if not access_code:
            access_code = generate_access_code()
            
        if not interview_date:
            flash('La date et l\'heure d\'entretien sont obligatoires!', 'danger')
            return render_template('admin_schedule_interview.html', application=application_data)
        
        try:
            # Convertir la date et l'heure au format lisible
            formatted_date = interview_date
            try:
                # Si la date est au format datetime-local (YYYY-MM-DDThh:mm)
                if 'T' in interview_date:
                    datetime_obj = datetime.strptime(interview_date, '%Y-%m-%dT%H:%M')
                    formatted_date = datetime_obj.strftime('%A %d %B %Y à %H:%M')
            except:
                pass
            
            # Récupérer les informations nécessaires
            job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], 
                                  application_data['category'], 
                                  f"{application_data['job_id']}.json")
            
            if os.path.exists(job_path):
                with open(job_path, 'r', encoding='utf-8') as f:
                    job_data = json.load(f)
            else:
                job_data = {'title': application_data['job_title']}
            
            # Préserver le score existant
            score = application_data.get('score')
            if not score and 'match_result' in application_data and 'overall_match' in application_data['match_result']:
                score = application_data['match_result']['overall_match']
            
            # Générer l'email d'invitation à l'entretien si aucun message personnalisé n'est fourni
            if not message:
                # Déterminer le type d'entretien
                interview_type_text = ""
                if interview_type == "video":
                    interview_type_text = f"en visioconférence via le lien suivant : {interview_location}"
                    
                    # Ajouter les informations sur l'entretien virtuel
                    interview_type_text += f"""
                    
                    Pour accéder à notre plateforme d'entretien virtuel, veuillez utiliser le code d'accès unique suivant :
                    CODE D'ACCÈS : {access_code}
                    
                    URL de l'entretien virtuel : {request.host_url}virtual_interview
                    """
                    
                elif interview_type == "phone":
                    interview_type_text = "par téléphone. Nous vous appellerons au numéro que vous nous avez communiqué"
                else:  # in person
                    interview_type_text = f"en personne à l'adresse suivante : {interview_location}"
                
                email_content = f"""
                Objet: Invitation à un entretien pour {job_data['title']}

                Bonjour {application_data['name']},

                Suite à l'examen de votre candidature pour le poste de {job_data['title']}, nous avons le plaisir de vous inviter à un entretien qui se tiendra le {formatted_date}.

                Cet entretien se déroulera {interview_type_text}.

                L'entretien durera environ 45 minutes et se déroulera comme suit :
                - Présentation de l'entreprise et du poste (10 minutes)
                - Discussion sur votre parcours et vos compétences (20 minutes)
                - Questions-réponses (15 minutes)

                Veuillez confirmer votre disponibilité pour cet entretien en répondant à cet e-mail.

                Si cette date ne vous convient pas, n'hésitez pas à nous proposer d'autres créneaux et nous ferons notre possible pour nous adapter.

                Nous vous recommandons de :
                - Relire la description du poste avant l'entretien
                - Préparer des questions sur le poste ou l'entreprise
                - Vous assurer que votre équipement est fonctionnel (pour un entretien à distance)

                Nous nous réjouissons de vous rencontrer et d'en apprendre davantage sur votre profil.

                Cordialement,
                L'équipe de recrutement
                """
            else:
                email_content = message
            
            # Envoyer l'email
            cv_parser.send_email(
                application_data['email'],
                f"Invitation à un entretien pour {job_data['title']}",
                email_content
            )
            
            # Mettre à jour le statut de la candidature
            application_data['status'] = 'interview'
            application_data['interview_date'] = formatted_date
            application_data['interview_type'] = interview_type
            application_data['interview_location'] = interview_location
            application_data['interview_email'] = email_content
            application_data['interview_scheduled_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            application_data['access_code'] = access_code
            
            # S'assurer que le score est préservé
            if score:
                application_data['score'] = score
            
            with open(application_path, 'w', encoding='utf-8') as f:
                json.dump(application_data, f, ensure_ascii=False, indent=4)
            
            flash('Invitation à l\'entretien envoyée avec succès!', 'success')
            return redirect(url_for('admin_application_detail', application_id=application_id))
        except Exception as e:
            flash(f'Erreur lors de l\'envoi de l\'email: {str(e)}', 'danger')
    
    return render_template('admin_schedule_interview.html', application=application_data)

@app.route('/admin/select_candidate_for_interview')
def admin_select_candidate_for_interview():
    """Page de sélection d'un candidat pour un entretien."""
    if not is_admin():
        flash('Accès restreint! Veuillez vous connecter.', 'danger')
        return redirect(url_for('login'))
    
    applications = get_applications()
    eligible_applications = [app for app in applications if app.get('status') in ['pending', 'accepted']]
    
    return render_template('admin_select_candidate.html', 
                          applications=eligible_applications, 
                          categories=CATEGORIES)

# ===== DÉBUT ROUTES DU CHATBOT =====

@app.route('/chatbot')
def chatbot():
    """Page de l'interface du chatbot."""
    return render_template('chatbot.html')

@app.route('/chatbot/initialize', methods=['GET'])
def chatbot_initialize():
    """Initialise le chatbot."""
    try:
        ensure_directories()
        get_documents_and_faiss_index()
        get_embedder()
        return jsonify({"status": "success", "message": "Chatbot initialized successfully"})
    except Exception as e:
        app.logger.error(f"Erreur lors de l'initialisation du chatbot: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/chatbot/query', methods=['POST'])
def chatbot_query():
    """Traite une requête textuelle pour le chatbot."""
    try:
        return process_query()
    except Exception as e:
        app.logger.error(f"Erreur lors du traitement de la requête: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Mise à jour du traitement audio pour résoudre l'erreur de PySoundFile
@app.route('/chatbot/query/audio', methods=['POST'])
def chatbot_query_audio():
    """Traite une requête audio pour le chatbot."""
    import time  # Ensure time module is available
    
    try:
        if 'audio' not in request.files:
            return jsonify({"status": "error", "message": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"status": "error", "message": "No audio file selected"}), 400
        
        # Créer un répertoire temporaire pour les fichiers audio si nécessaire
        temp_dir = os.path.join(os.getcwd(), 'temp_audio')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Ensure we use the correct file extension based on content
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        if not file_ext:
            # Default to .webm if no extension is provided
            file_ext = '.webm'
        
        # Save the file with correct extension
        timestamp = int(time.time())
        original_filename = f"recording_{timestamp}{file_ext}"
        original_path = os.path.join(temp_dir, original_filename)
        audio_file.save(original_path)
        
        app.logger.info(f"Saved audio file: {original_path}")
        
        # For now, return a simple response since audio processing is challenging
        transcription = "Votre message audio a été reçu"
        answer = "J'ai bien reçu votre message audio. Je travaille toujours sur l'amélioration de la reconnaissance vocale. Pourriez-vous taper votre question pour le moment?"
        
        # Clean up temporary file
        try:
            if os.path.exists(original_path):
                os.remove(original_path)
        except Exception as e:
            app.logger.warning(f"Could not delete temp file: {str(e)}")
        
        return jsonify({
            "status": "success",
            "transcription": transcription,
            "answer": answer
        })
        
    except Exception as e:
        import traceback
        app.logger.error(f"Error processing audio: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/chatbot/upload/document', methods=['POST'])
def chatbot_upload_document():
    """Télécharge et indexe un document pour le chatbot."""
    try:
        if not is_admin():
            return jsonify({"status": "error", "message": "Accès restreint aux administrateurs"}), 403
        return upload_document()
    except Exception as e:
        app.logger.error(f"Erreur lors de l'upload du document: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/chatbot/agents', methods=['GET'])
def chatbot_agents():
    """Liste les agents disponibles du chatbot."""
    try:
        return list_agents()
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération des agents: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/chatbot/query/agent', methods=['POST'])
def chatbot_query_agent():
    """Interroge un agent spécifique du chatbot."""
    try:
        return query_specific_agent()
    except Exception as e:
        app.logger.error(f"Erreur lors de l'interrogation de l'agent: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ===== FIN ROUTES DU CHATBOT =====

# ===== DÉBUT ROUTES DE L'ENTRETIEN VIRTUEL =====

@app.route('/virtual_interview')
def virtual_interview():
    """Page d'entretien virtuel."""
    return render_template('virtual_interview.html')

@app.route('/verify-interview-access', methods=['POST'])
def verify_interview_access():
    """Vérifie les informations d'accès pour l'entretien virtuel."""
    data = request.json
    email = data.get('email')
    access_code = data.get('access_code')
    
    if not email or not access_code:
        return jsonify({
            'status': 'error',
            'message': 'Veuillez fournir un email et un code d\'accès.'
        }), 400
    
    # Vérifier si les informations d'accès correspondent à une candidature
    applications = get_applications()
    
    for application in applications:
        # Vérifier si l'email et le code d'accès correspondent
        if (application.get('email') == email and 
            application.get('access_code') == access_code and 
            application.get('status') == 'interview'):
            
            # Succès : retourner les informations du candidat
            return jsonify({
                'status': 'success',
                'message': 'Authentification réussie',
                'candidate_info': {
                    'name': application.get('name'),
                    'job_title': application.get('job_title'),
                    'interview_date': application.get('interview_date'),
                }
            })
    
    # Échec : aucune correspondance trouvée
    return jsonify({
        'status': 'error',
        'message': 'Code d\'accès ou email invalide. Veuillez vérifier vos informations et réessayer.'
    }), 401

# Définition de la route pour démarrer l'entretien
@app.route('/start-interview', methods=['POST'])
def start_interview():
    """Initialise une nouvelle session d'entretien"""
    global interviewer
    
    data = request.json
    candidate_name = data.get('candidateName', '')
    position = data.get('position', '')
    domains = data.get('domains', [])
    email = data.get('email', '')
    access_code = data.get('accessCode', '')
    
    # Vérifier les informations d'accès
    valid_credentials = False
    application_data = None
    
    if email and access_code:
        applications = get_applications()
        
        for app_data in applications:
            if (app_data.get('email') == email and 
                app_data.get('access_code') == access_code and 
                app_data.get('status') == 'interview'):
                valid_credentials = True
                application_data = app_data
                break
    
    if not valid_credentials:
        return jsonify({
            'status': 'error',
            'message': 'Accès non autorisé. Veuillez vous authentifier à nouveau.'
        }), 401

    try:
        # Création de l'interviewer
        interviewer = AITechnicalInterviewer()
        
        # Enregistrer l'heure de début de l'entretien dans le fichier d'application
        if application_data and 'id' in application_data:
            try:
                # Utiliser la variable app de Flask, pas l'objet application_data
                application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_data['id']}.json")
                
                if os.path.exists(application_path):
                    with open(application_path, 'r', encoding='utf-8') as f:
                        app_data = json.load(f)
                    
                    app_data['interview_started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    with open(application_path, 'w', encoding='utf-8') as f:
                        json.dump(app_data, f, ensure_ascii=False, indent=4)
            except Exception as file_error:
                # Utiliser app.logger
                app.logger.error(f"Erreur lors de la mise à jour du fichier d'application: {str(file_error)}")
                # Continuer malgré l'erreur, car ce n'est pas critique
        
        # Démarrer l'entretien dans un thread séparé
        try:
            interview_thread = threading.Thread(
                target=interviewer.start_interview,
                args=(candidate_name, position, domains)
            )
            interview_thread.daemon = True
            interview_thread.start()
        except Exception as thread_error:
            app.logger.error(f"Erreur lors du démarrage du thread d'entretien: {str(thread_error)}")
            return jsonify({'status': 'error', 'message': f"Erreur lors du démarrage de l'entretien: {str(thread_error)}"})

        return jsonify({'status': 'success', 'message': 'Entretien démarré'})
    except Exception as e:
        app.logger.error(f"Erreur globale lors du démarrage de l'entretien: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# Les gestionnaires d'événements Socket.IO doivent être définis au niveau global
@socketio.on('video_frame')
def handle_video_frame(frame_data):
    """Gère la réception des frames vidéo"""
    try:
        face_detected = process_video_frame(frame_data)
        video_queue.put(face_detected)
    except Exception as e:
        print(f"Erreur dans handle_video_frame: {e}")

@socketio.on('audio_data')
def handle_audio_data(audio_data):
    """Gère la réception des données audio"""
    if interviewer:
        try:
            response = interviewer.process_audio(audio_data)
            if response:
                emit('interviewer_response', {'text': response})
                
            # Supposons que la transcription est disponible via interviewer.get_transcription()
            transcription = interviewer.get_transcription() if hasattr(interviewer, 'get_transcription') else None
            if transcription:
                emit('audio_transcription', {'text': transcription})
        except Exception as e:
            print(f"Erreur dans handle_audio_data: {e}")
            emit('system_message', {'text': f'Erreur de traitement audio: {str(e)}'})

@socketio.on('text_message')
def handle_text_message(message):
    """Gère la réception des messages texte"""
    if interviewer:
        try:
            response = interviewer.process_text(message)
            if response:
                emit('interviewer_response', {'text': response})
        except Exception as e:
            print(f"Erreur dans handle_text_message: {e}")
            emit('system_message', {'text': f'Erreur de traitement du message: {str(e)}'})

@socketio.on('connect')
def handle_connect():
    """Gère la connexion d'un client"""
    print('Client connecté')
    emit('system_message', {'text': 'Connexion établie'})

@socketio.on('disconnect')
def handle_disconnect():
    """Gère la déconnexion d'un client"""
    print('Client déconnecté')
    
# ===== FIN ROUTES DE L'ENTRETIEN VIRTUEL =====

@app.errorhandler(404)
def page_not_found(e):
    """Gestion des erreurs 404."""
    path = request.path
    
    # Journaliser l'erreur
    app.logger.warning(f'Page non trouvée: {path}')
    
    # Gestion spécifique pour des routes courantes avec slash manquant ou de trop
    if path.endswith('/'):
        alternative_path = path[:-1]  # Enlever le slash final
    else:
        alternative_path = f"{path}/"  # Ajouter un slash final
    
    # Essayer de rediriger vers la route alternative si elle existe
    try:
        for rule in app.url_map.iter_rules():
            if rule.rule == alternative_path:
                return redirect(alternative_path)
    except:
        pass
    
    # CORRECTION: Modifier la condition pour éviter les faux positifs
    # Gestion spéciale pour certaines routes
    if 'admin/application' in path and not any(c.isdigit() for c in path) and 'delete' not in path:
        if path == '/admin/application' or path == '/admin/application/':
            flash('Veuillez sélectionner une candidature spécifique à consulter.', 'warning')
            return redirect(url_for('admin_dashboard'))
    
    if 'admin/schedule_interview' in path and not any(c.isdigit() for c in path):
        if not path.endswith('admin/select_candidate_for_interview'):
            flash('Veuillez sélectionner une candidature pour programmer un entretien.', 'warning')
            return redirect(url_for('admin_select_candidate_for_interview'))
        
    if 'admin/application/accept' in path and not any(c.isdigit() for c in path):
        flash('Veuillez sélectionner une candidature pour l\'accepter.', 'warning')
        return redirect(url_for('admin_dashboard'))
        
    if 'admin/application/reject' in path and not any(c.isdigit() for c in path):
        flash('Veuillez sélectionner une candidature pour la rejeter.', 'warning')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('404.html'), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Gestion des erreurs 405 - Method Not Allowed."""
    app.logger.error(f"Method Not Allowed: {request.method} {request.path}")
    flash(f'Méthode {request.method} non autorisée pour cette URL.', 'danger')
    return render_template('405.html', error=e), 405

@app.errorhandler(500)
def internal_server_error(e):
    """Gestion des erreurs 500."""
    app.logger.error(f"Erreur interne du serveur: {str(e)}")
    return render_template('500.html'), 500

# Route de diagnostic pour déboguer les problèmes de routes
@app.route('/debug/routes')
def debug_routes():
    """Affiche toutes les routes enregistrées pour le débogage."""
    if not is_admin():
        return abort(403)
    
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': ','.join(rule.methods),
            'rule': rule.rule
        })
    
    return jsonify({'routes': routes})

# Initialiser l'interviewer AI
interviewer = None
video_queue = queue.Queue()
user_present = False

# Configuration de MediaPipe pour la détection faciale
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def process_video_frame(frame_data):
    global frame_counter
    frame_counter += 1

    # Ne traite la détection que sur 1 frame sur 3
    if frame_counter % 3 != 0:
        # Optionnel : renvoyer la frame brute sans détection
        try:
            encoded_data = frame_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            _, buffer = cv2.imencode('.jpg', frame)
            modified_frame_b64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('processed_frame', {
                'frame_b64': modified_frame_b64,
                'detection_info': {
                    'face_detected': False,
                    'phone_detected': False,
                    'emotion_detected': False,
                    'phone_conf': 0,
                    'emotion': '',
                    'mapped_emotion': '',
                    'emotion_conf': 0
                }
            })
        except Exception as e:
            print(f"Erreur lors du traitement de la frame ignorée: {e}")
        return False

    # --- Le reste de ta fonction process_video_frame (détection complète) ---
    try:
        # Décoder l'image base64
        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convertir en RGB pour MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        modified_frame = frame.copy()
        detection_info = {
            'face_detected': False,
            'phone_detected': False,
            'emotion_detected': False,
            'phone_conf': 0,
            'emotion': '',
            'mapped_emotion': '',
            'emotion_conf': 0
        }

        # --- Détection de téléphone ---
        phone_detected, phone_conf = detect_phone_in_frame(phone_model, frame)
        if phone_detected:
            height, width = frame.shape[:2]
            cv2.rectangle(modified_frame, (10, 10), (width-10, height-10), (0, 0, 255), 10)
            cv2.putText(modified_frame, f"TELEPHONE DETECTE! ({phone_conf:.2f})", 
                       (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            save_fraud_detection_image(frame, phone_conf)
            detection_info['phone_detected'] = True
            detection_info['phone_conf'] = phone_conf

        # --- Détection d'émotion ---
        emotion, mapped_emotion, emo_conf = detect_emotion_in_frame(emotion_model, frame)
        if emotion:
            cv2.putText(modified_frame, f"Emotion: {emotion}/{mapped_emotion} ({emo_conf:.2f})", 
                       (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            save_emotion_detection_image(frame, emotion, mapped_emotion, emo_conf)
            detection_info['emotion_detected'] = True
            detection_info['emotion'] = emotion
            detection_info['mapped_emotion'] = mapped_emotion
            detection_info['emotion_conf'] = float(emo_conf)

        # --- Détection de présence (visage) ---
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = modified_frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(modified_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            detection_info['face_detected'] = True

        # Convertir la frame modifiée en base64 pour l'envoi
        _, buffer = cv2.imencode('.jpg', modified_frame)
        modified_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('processed_frame', {
            'frame_b64': modified_frame_b64,
            'detection_info': detection_info
        })

        return detection_info['face_detected']
    except Exception as e:
        print(f"Erreur lors du traitement de la frame: {e}")
        return False

def monitor_user_presence():
    """Surveille la présence de l'utilisateur via la détection faciale"""
    global user_present
    while True:
        try:
            # Attendre jusqu'à 3 secondes pour un résultat
            face_detected = video_queue.get(timeout=3)
            
            # Mettre à jour l'état de présence de l'utilisateur
            if face_detected != user_present:
                user_present = face_detected
                if interviewer:
                    if user_present:
                        emit('system_message', {'text': 'Utilisateur détecté'}, broadcast=True)
                    else:
                        emit('system_message', {'text': 'Utilisateur non détecté'}, broadcast=True)
                        
        except queue.Empty:
            # Timeout, aucune frame n'a été reçue
            if user_present:
                user_present = False
                if interviewer:
                    emit('system_message', {'text': 'Aucune détection faciale depuis 3 secondes'}, broadcast=True)
        except Exception as e:
            print(f"Erreur dans le thread de surveillance: {e}")
            time.sleep(1)  # Éviter une boucle infinie d'erreurs

# Démarrer le thread de surveillance
presence_thread = threading.Thread(target=monitor_user_presence, daemon=True)
presence_thread.start()

# Fonction pour journaliser les événements de l'entretien
def log_interview_event(application_id, event_type, description):
    """
    Journalise un événement d'entretien dans le fichier de candidature.
    
    :param application_id: ID de la candidature
    :param event_type: Type d'événement (ex: 'connection', 'phone_detection', 'emotion', 'disconnection')
    :param description: Description détaillée de l'événement
    """
    try:
        application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_id}.json")
        if os.path.exists(application_path):
            with open(application_path, 'r', encoding='utf-8') as f:
                app_data = json.load(f)
            
            # Créer la structure pour les logs d'entretien si elle n'existe pas
            if 'interview_logs' not in app_data:
                app_data['interview_logs'] = []
            
            # Ajouter le nouvel événement
            event = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': event_type,
                'description': description
            }
            app_data['interview_logs'].append(event)
            
            # Sauvegarder les modifications
            with open(application_path, 'w', encoding='utf-8') as f:
                json.dump(app_data, f, ensure_ascii=False, indent=4)
                
    except Exception as e:
        app.logger.error(f"Erreur lors de la journalisation de l'événement d'entretien: {str(e)}")

if __name__ == '__main__':
    # Configurer la journalisation
    logging.basicConfig(
        filename='app.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # S'assurer que les dépendances pour le traitement audio sont installées
    try:
        import soundfile as sf
        app.logger.info("SoundFile library is available")
    except ImportError:
        app.logger.warning("SoundFile not installed. Installing required dependencies is recommended.")
    
    # Lancer l'application avec Socket.IO
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)