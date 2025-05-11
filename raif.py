import os
import json
import secrets
from datetime import datetime, timedelta
import jwt
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, jsonify, send_from_directory, abort
from flask_cors import CORS
import logging

# Importer le parser de CV
from main import CVParser

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
# Activer CORS pour permettre à Angular de communiquer avec Flask
CORS(app)

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
app.config['JWT_SECRET_KEY'] = secrets.token_hex(32)
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

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

def generate_token(user_id):
    """Génère un token JWT pour l'authentification."""
    expiration = datetime.utcnow() + app.config['JWT_ACCESS_TOKEN_EXPIRES']
    payload = {
        'user_id': user_id,
        'exp': expiration
    }
    return jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')

def verify_token(token):
    """Vérifie un token JWT."""
    try:
        payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# API Routes

@app.route('/api/login', methods=['POST'])
def api_login():
    """Endpoint d'authentification."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if username == app.config['ADMIN_USERNAME'] and password == app.config['ADMIN_PASSWORD']:
        token = generate_token(username)
        return jsonify({'token': token, 'message': 'Connexion réussie'})
    else:
        return jsonify({'message': 'Nom d\'utilisateur ou mot de passe incorrect'}), 401

@app.route('/api/categories', methods=['GET'])
def api_categories():
    """Renvoie la liste des catégories d'emploi."""
    return jsonify(CATEGORIES)

@app.route('/api/jobs', methods=['GET'])
def api_jobs():
    """Renvoie toutes les offres d'emploi."""
    all_jobs = []
    
    for category, category_name in CATEGORIES.items():
        jobs = get_job_offers(category)
        for job in jobs:
            job['category_name'] = category_name
        all_jobs.extend(jobs)
    
    return jsonify(all_jobs)

@app.route('/api/jobs/<category>', methods=['GET'])
def api_jobs_by_category(category):
    """Renvoie les offres d'emploi d'une catégorie."""
    if category not in CATEGORIES:
        return jsonify({'error': 'Catégorie non valide'}), 400
    
    jobs = get_job_offers(category)
    return jsonify(jobs)

@app.route('/api/jobs/<category>/<job_id>', methods=['GET'])
def api_job_detail(category, job_id):
    """Renvoie les détails d'une offre d'emploi."""
    job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], category, f"{job_id}.json")
    
    if not os.path.exists(job_path):
        return jsonify({'error': 'Offre d\'emploi non trouvée'}), 404
    
    with open(job_path, 'r', encoding='utf-8') as f:
        job_data = json.load(f)
    
    job_data['id'] = job_id
    return jsonify(job_data)

@app.route('/api/jobs', methods=['POST'])
def api_add_job():
    """Ajoute une nouvelle offre d'emploi."""
    # Vérifier le token
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Token non fourni'}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    if not payload:
        return jsonify({'error': 'Token invalide ou expiré'}), 401
    
    data = request.get_json()
    job_title = data.get('title')
    job_description = data.get('description')
    category = data.get('category')
    
    if not job_title or not job_description or category not in CATEGORIES:
        return jsonify({'error': 'Données manquantes ou catégorie invalide'}), 400
    
    job_id = f"{int(datetime.now().timestamp())}"
    job_data = {
        'title': job_title,
        'description': job_description,
        'category': category,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    category_folder = os.path.join(app.config['JOB_OFFERS_FOLDER'], category)
    os.makedirs(category_folder, exist_ok=True)
    
    with open(os.path.join(category_folder, f"{job_id}.json"), 'w', encoding='utf-8') as f:
        json.dump(job_data, f, ensure_ascii=False, indent=4)
    
    job_data['id'] = job_id
    return jsonify(job_data), 201

@app.route('/api/jobs/<category>/<job_id>', methods=['PUT'])
def api_update_job(category, job_id):
    """Met à jour une offre d'emploi."""
    # Vérifier le token
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Token non fourni'}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    if not payload:
        return jsonify({'error': 'Token invalide ou expiré'}), 401
    
    job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], category, f"{job_id}.json")
    
    if not os.path.exists(job_path):
        return jsonify({'error': 'Offre d\'emploi non trouvée'}), 404
    
    with open(job_path, 'r', encoding='utf-8') as f:
        job_data = json.load(f)
    
    data = request.get_json()
    
    # Mettre à jour les données
    job_data['title'] = data.get('title', job_data['title'])
    job_data['description'] = data.get('description', job_data['description'])
    job_data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(job_path, 'w', encoding='utf-8') as f:
        json.dump(job_data, f, ensure_ascii=False, indent=4)
    
    job_data['id'] = job_id
    return jsonify(job_data)

@app.route('/api/jobs/<category>/<job_id>', methods=['DELETE'])
def api_delete_job(category, job_id):
    """Supprime une offre d'emploi."""
    # Vérifier le token
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Token non fourni'}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    if not payload:
        return jsonify({'error': 'Token invalide ou expiré'}), 401
    
    job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], category, f"{job_id}.json")
    
    if not os.path.exists(job_path):
        return jsonify({'error': 'Offre d\'emploi non trouvée'}), 404
    
    # Vérifier si des candidatures sont liées à cette offre
    linked_applications = []
    for filename in os.listdir(app.config['APPLICATIONS_FOLDER']):
        if filename.endswith('.json'):
            with open(os.path.join(app.config['APPLICATIONS_FOLDER'], filename), 'r', encoding='utf-8') as f:
                try:
                    application = json.load(f)
                    if application.get('job_id') == job_id and application.get('category') == category:
                        application['id'] = filename.split('.')[0]
                        linked_applications.append(application)
                except json.JSONDecodeError:
                    continue
    
    # Supprimer l'offre d'emploi
    os.remove(job_path)
    
    # Supprimer toutes les candidatures associées
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
    
    return jsonify({
        'message': 'Offre d\'emploi supprimée avec succès',
        'deleted_applications': deleted_count
    })

@app.route('/api/applications', methods=['GET'])
def api_applications():
    """Renvoie toutes les candidatures."""
    # Vérifier le token
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Token non fourni'}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    if not payload:
        return jsonify({'error': 'Token invalide ou expiré'}), 401
    
    applications = get_applications()
    
    # Filtrage par statut si demandé
    status_filter = request.args.get('status')
    if status_filter:
        applications = [app for app in applications if app.get('status') == status_filter]
    
    # Tri des résultats
    sort_option = request.args.get('sort', 'score')
    if sort_option == 'date':
        applications.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    elif sort_option == 'alpha':
        applications.sort(key=lambda x: x.get('name', '').lower())
    
    return jsonify(applications)

@app.route('/api/applications/<application_id>', methods=['GET'])
def api_application_detail(application_id):
    """Renvoie les détails d'une candidature."""
    # Vérifier le token
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Token non fourni'}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    if not payload:
        return jsonify({'error': 'Token invalide ou expiré'}), 401
    
    application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_id}.json")
    
    if not os.path.exists(application_path):
        return jsonify({'error': 'Candidature non trouvée'}), 404
    
    with open(application_path, 'r', encoding='utf-8') as f:
        application_data = json.load(f)
    
    application_data['id'] = application_id
    return jsonify(application_data)

@app.route('/api/applications', methods=['POST'])
def api_submit_application():
    """Soumet une nouvelle candidature."""
    # Pas besoin de token ici, c'est une route publique
    
    # Vérifier si le formulaire contient les champs requis
    if 'name' not in request.form or 'email' not in request.form or 'job_id' not in request.form or 'category' not in request.form:
        return jsonify({'error': 'Données du formulaire manquantes'}), 400
    
    name = request.form.get('name')
    email = request.form.get('email')
    job_id = request.form.get('job_id')
    category = request.form.get('category')
    
    # Vérifier si la catégorie est valide
    if category not in CATEGORIES:
        return jsonify({'error': 'Catégorie invalide'}), 400
    
    # Vérifier si l'offre d'emploi existe
    job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], category, f"{job_id}.json")
    if not os.path.exists(job_path):
        return jsonify({'error': 'Offre d\'emploi non trouvée'}), 404
    
    with open(job_path, 'r', encoding='utf-8') as f:
        job_data = json.load(f)
    
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
        return jsonify({'error': 'Vous avez déjà postulé à cette offre avec cette adresse email'}), 400
    
    # Vérifier si le fichier a été soumis
    if 'cv_file' not in request.files:
        return jsonify({'error': 'Aucun fichier CV trouvé'}), 400
    
    file = request.files['cv_file']
    
    # Si l'utilisateur ne sélectionne pas de fichier
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
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
        
        application_data['id'] = application_id
        return jsonify({
            'message': 'Candidature soumise avec succès',
            'application': application_data
        }), 201
    else:
        return jsonify({'error': 'Type de fichier non autorisé! Seuls les fichiers PDF sont acceptés'}), 400

@app.route('/api/applications/<application_id>/analyze', methods=['POST'])
def api_analyze_application(application_id):
    """Analyse une candidature avec le parser de CV."""
    # Vérifier le token
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Token non fourni'}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    if not payload:
        return jsonify({'error': 'Token invalide ou expiré'}), 401
    
    application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_id}.json")
    
    if not os.path.exists(application_path):
        return jsonify({'error': 'Candidature non trouvée'}), 404
    
    with open(application_path, 'r', encoding='utf-8') as f:
        application_data = json.load(f)
    
    # Récupérer le chemin du fichier CV
    cv_path = os.path.join(app.config['UPLOAD_FOLDER'], application_data['cv_file'])
    
    if not os.path.exists(cv_path):
        return jsonify({'error': 'Fichier CV non trouvé'}), 404
    
    try:
        # Analyser le CV
        cv_info = cv_parser.parse_cv(cv_path)
        
        # Récupérer l'offre d'emploi associée
        job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], 
                               application_data['category'], 
                               f"{application_data['job_id']}.json")
        
        if not os.path.exists(job_path):
            return jsonify({'error': 'Offre d\'emploi associée non trouvée'}), 404
        
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
        
        return jsonify({
            'message': 'Analyse du CV réussie',
            'cv_info': cv_info,
            'match_result': match_result
        })
    except Exception as e:
        return jsonify({'error': f'Erreur lors de l\'analyse du CV: {str(e)}'}), 500

@app.route('/api/applications/<application_id>/status', methods=['PUT'])
def api_update_application_status(application_id):
    """Met à jour le statut d'une candidature."""
    # Vérifier le token
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Token non fourni'}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    if not payload:
        return jsonify({'error': 'Token invalide ou expiré'}), 401
    
    application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_id}.json")
    
    if not os.path.exists(application_path):
        return jsonify({'error': 'Candidature non trouvée'}), 404
    
    with open(application_path, 'r', encoding='utf-8') as f:
        application_data = json.load(f)
    
    data = request.get_json()
    status = data.get('status')
    
    if not status or status not in ['pending', 'accepted', 'rejected', 'interview']:
        return jsonify({'error': 'Statut invalide'}), 400
    
    # Mettre à jour le statut
    application_data['status'] = status
    application_data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Traitement spécifique selon le statut
    if status == 'accepted':
        # Gérer l'acceptation
        acceptance_message = data.get('message', '')
        
        if acceptance_message:
            application_data['response_email'] = acceptance_message
            
            # Envoyer l'email d'acceptation
            try:
                job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], 
                                      application_data['category'], 
                                      f"{application_data['job_id']}.json")
                
                with open(job_path, 'r', encoding='utf-8') as f:
                    job_data = json.load(f)
                
                cv_parser.send_email(
                    application_data['email'],
                    f"Candidature retenue pour {job_data['title']}",
                    acceptance_message
                )
                
                application_data['response_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                return jsonify({'error': f'Erreur lors de l\'envoi de l\'email: {str(e)}'}), 500
    
    elif status == 'rejected':
        # Gérer le rejet
        rejection_message = data.get('message', '')
        
        if rejection_message:
            application_data['response_email'] = rejection_message
            
            # Envoyer l'email de rejet
            try:
                job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], 
                                      application_data['category'], 
                                      f"{application_data['job_id']}.json")
                
                with open(job_path, 'r', encoding='utf-8') as f:
                    job_data = json.load(f)
                
                cv_parser.send_email(
                    application_data['email'],
                    f"Réponse concernant votre candidature pour {job_data['title']}",
                    rejection_message
                )
                
                application_data['response_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                return jsonify({'error': f'Erreur lors de l\'envoi de l\'email: {str(e)}'}), 500
    
    elif status == 'interview':
        # Gérer la programmation d'entretien
        interview_date = data.get('interview_date', '')
        interview_link = data.get('interview_link', '')
        interview_message = data.get('message', '')
        
        if interview_date and interview_link and interview_message:
            application_data['interview_date'] = interview_date
            application_data['interview_link'] = interview_link
            application_data['interview_email'] = interview_message
            
            # Envoyer l'email d'invitation à l'entretien
            try:
                job_path = os.path.join(app.config['JOB_OFFERS_FOLDER'], 
                                      application_data['category'], 
                                      f"{application_data['job_id']}.json")
                
                with open(job_path, 'r', encoding='utf-8') as f:
                    job_data = json.load(f)
                
                cv_parser.send_email(
                    application_data['email'],
                    f"Invitation à un entretien pour {job_data['title']}",
                    interview_message
                )
                
                application_data['interview_scheduled_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                return jsonify({'error': f'Erreur lors de l\'envoi de l\'email: {str(e)}'}), 500
    
    # Enregistrer les modifications
    with open(application_path, 'w', encoding='utf-8') as f:
        json.dump(application_data, f, ensure_ascii=False, indent=4)
    
    application_data['id'] = application_id
    return jsonify(application_data)

@app.route('/api/applications/<application_id>/notes', methods=['PUT'])
def api_update_application_notes(application_id):
    """Met à jour les notes d'une candidature."""
    # Vérifier le token
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Token non fourni'}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    if not payload:
        return jsonify({'error': 'Token invalide ou expiré'}), 401
    
    application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_id}.json")
    
    if not os.path.exists(application_path):
        return jsonify({'error': 'Candidature non trouvée'}), 404
    
    with open(application_path, 'r', encoding='utf-8') as f:
        application_data = json.load(f)
    
    data = request.get_json()
    notes = data.get('notes', '')
    
    # Mettre à jour les notes
    application_data['admin_notes'] = notes
    
    # Enregistrer les modifications
    with open(application_path, 'w', encoding='utf-8') as f:
        json.dump(application_data, f, ensure_ascii=False, indent=4)
    
    application_data['id'] = application_id
    return jsonify(application_data)

@app.route('/api/applications/<application_id>', methods=['DELETE'])
def api_delete_application(application_id):
    """Supprime une candidature."""
    # Vérifier le token
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Token non fourni'}), 401
    
    token = auth_header.split(' ')[1]
    payload = verify_token(token)
    if not payload:
        return jsonify({'error': 'Token invalide ou expiré'}), 401
    
    application_path = os.path.join(app.config['APPLICATIONS_FOLDER'], f"{application_id}.json")
    
    if not os.path.exists(application_path):
        return jsonify({'error': 'Candidature non trouvée'}), 404
    
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
    
    return jsonify({'message': 'Candidature supprimée avec succès'})

@app.route('/api/uploads/<filename>')
def api_download_file(filename):
    """Télécharge un fichier."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Point d'entrée pour servir l'application Angular
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Route pour servir l'application Angular."""
    # Cette route sert les fichiers statiques de l'application Angular
    # En production, vous voudrez peut-être configurer un serveur web comme Nginx pour servir ces fichiers
    
    # Pour le développement, on peut simplement renvoyer un message
    return jsonify({'message': 'API backend en cours d\'exécution. Veuillez démarrer le frontend Angular séparément.'})

if __name__ == '__main__':
    # Configurer la journalisation
    logging.basicConfig(
        filename='app.log',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app.run(debug=True)