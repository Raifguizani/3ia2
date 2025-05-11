import os
import json
import time
import uuid
import hashlib
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, abort
import logging
import traceback
from functools import wraps
from datetime import datetime
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "votre_cle_secrete_tres_longue_et_aleatoire")  # Utiliser la clé du .env si disponible
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # Session d'une heure

# Pour utiliser datetime dans les templates
app.jinja_env.globals.update(now=datetime.now)

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration des uploads
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'recruitment_data'
JOB_FOLDER = 'job_descriptions'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Créer les répertoires nécessaires
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(JOB_FOLDER, exist_ok=True)

# Identifiants admin - en production, utilisez une base de données avec des mots de passe hashés
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")

# Fonctions d'authentification
def is_authenticated():
    """Simple check for authentication"""
    return session.get('logged_in', False)

def login_required(f):
    """Décorateur pour protéger les routes admin"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            flash('Vous devez vous connecter d\'abord.', 'danger')
            return redirect(url_for('admin_login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    """Vérifie si l'extension du fichier est autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fonctions auxiliaires pour les descriptions de postes
def get_all_job_descriptions():
    """Récupère toutes les descriptions de postes"""
    job_descriptions = []
    try:
        for filename in os.listdir(JOB_FOLDER):
            if filename.endswith('.json'):
                with open(os.path.join(JOB_FOLDER, filename), 'r', encoding='utf-8') as f:
                    job_data = json.load(f)
                    job_data['id'] = filename.replace('.json', '')
                    job_descriptions.append(job_data)
        return sorted(job_descriptions, key=lambda x: x.get('created_at', ''), reverse=True)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des descriptions de postes: {str(e)}")
        return []

def get_job_description(job_id):
    """Récupère une description de poste spécifique"""
    try:
        with open(os.path.join(JOB_FOLDER, f"{job_id}.json"), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la description de poste {job_id}: {str(e)}")
        return None

def save_job_description(job_data):
    """Sauvegarde une description de poste"""
    try:
        job_id = job_data.get('id', str(uuid.uuid4()))
        job_data['id'] = job_id
        
        # Ajouter un horodatage de création si nouveau
        if 'created_at' not in job_data:
            job_data['created_at'] = datetime.now().isoformat()
        
        # Toujours mettre à jour l'horodatage de modification
        job_data['updated_at'] = datetime.now().isoformat()
        
        with open(os.path.join(JOB_FOLDER, f"{job_id}.json"), 'w', encoding='utf-8') as f:
            json.dump(job_data, f, ensure_ascii=False, indent=4)
        return job_id
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de la description de poste: {str(e)}")
        return None

def delete_job_description(job_id):
    """Supprime une description de poste"""
    try:
        job_file = os.path.join(JOB_FOLDER, f"{job_id}.json")
        if os.path.exists(job_file):
            os.remove(job_file)
            return True
        return False
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la description de poste {job_id}: {str(e)}")
        return False

# Ajouter un filtre nl2br personnalisé pour convertir les retours à la ligne en <br>
@app.template_filter('nl2br')
def nl2br(value):
    """Convert newlines to <br> tags."""
    if value:
        return value.replace('\n', '<br>\n')
    return value

# Routes publiques
@app.route('/')
def index():
    """Page d'accueil avec les offres d'emploi"""
    jobs = get_all_job_descriptions()
    # Afficher uniquement les offres publiées au public
    public_jobs = [job for job in jobs if job.get('status') == 'published']
    return render_template('index.html', jobs=public_jobs)

@app.route('/job/<job_id>')
def view_job(job_id):
    """Voir une offre spécifique et le formulaire de candidature"""
    job = get_job_description(job_id)
    if not job or job.get('status') != 'published':
        flash('Offre d\'emploi non trouvée ou plus disponible', 'warning')
        return redirect(url_for('index'))
    
    return render_template('job_apply.html', job=job)

@app.route('/apply/<job_id>', methods=['POST'])
def apply_job(job_id):
    """Gérer la soumission de candidature"""
    job = get_job_description(job_id)
    if not job or job.get('status') != 'published':
        flash('Offre d\'emploi non trouvée ou plus disponible', 'warning')
        return redirect(url_for('index'))
    
    # Vérifier si la requête contient un fichier
    if 'cv_file' not in request.files:
        flash('Aucun fichier sélectionné', 'danger')
        return redirect(url_for('view_job', job_id=job_id))
    
    file = request.files['cv_file']
    
    # Si l'utilisateur ne sélectionne pas de fichier
    if file.filename == '':
        flash('Aucun fichier sélectionné', 'danger')
        return redirect(url_for('view_job', job_id=job_id))
    
    if file and allowed_file(file.filename):
        try:
            # Créer un nom de fichier unique pour éviter les écrasements
            unique_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            base_name, extension = os.path.splitext(filename)
            unique_filename = f"{base_name}_{unique_id}{extension}"
            
            # Sauvegarder le fichier
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Stocker les informations dans la session
            session['file_path'] = file_path
            session['orig_filename'] = filename
            session['job_description'] = json.dumps(job)  # Stocker la description du poste dans la session
            session['processing_id'] = unique_id
            session['job_id'] = job_id
            
            logger.info(f"Fichier téléchargé pour le poste {job_id}: {filename} -> {file_path} avec ID {unique_id}")
            
            # Rediriger vers la page de traitement
            return redirect(url_for('processing'))
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement du fichier: {str(e)}")
            flash(f'Erreur lors du téléchargement du fichier: {str(e)}', 'danger')
            return redirect(url_for('view_job', job_id=job_id))
    
    flash('Type de fichier non valide. Veuillez télécharger un fichier PDF.', 'danger')
    return redirect(url_for('view_job', job_id=job_id))

@app.route('/processing')
def processing():
    """Affiche la page de traitement"""
    # Vérifier si nous avons des données dans la session
    if 'file_path' not in session or 'job_description' not in session:
        flash('Aucune donnée à traiter. Veuillez télécharger un CV et sélectionner un poste d\'abord.', 'warning')
        return redirect(url_for('index'))
    
    orig_filename = session.get('orig_filename', 'Fichier CV')
    return render_template(
        'processing.html', 
        processing_id=session.get('processing_id', ''),
        filename=orig_filename
    )

@app.route('/process_cv', methods=['POST'])
def process_cv():
    """Endpoint API pour traiter le CV et la description du poste"""
    # Récupérer les données de la session
    file_path = session.get('file_path')
    job_description = session.get('job_description')
    processing_id = session.get('processing_id')
    
    if not file_path or not job_description or not processing_id:
        logger.warning("Données manquantes dans la session pour le traitement")
        return jsonify({'success': False, 'error': 'Données requises manquantes'})
    
    try:
        logger.info(f"Démarrage du traitement pour ID: {processing_id}")
        
        # Traiter la candidature
        # La fonction process_application de main.py génère déjà les emails
        from main import process_application
        result = process_application(file_path, job_description)
        
        if "error" in result and not result.get("partial_success", False):
            logger.error(f"Erreur de traitement: {result['error']}")
            return jsonify({'success': False, 'error': result['error']})
        
        # Sauvegarder les résultats dans un fichier JSON pour référence future
        result_filename = f"{processing_id}_results.json"
        result_path = os.path.join(DATA_FOLDER, result_filename)
        
        # Ajouter job_id et processing_id aux résultats
        result['job_id'] = session.get('job_id')
        result['id'] = processing_id  # Important pour la suppression de candidat
        result['submission_date'] = datetime.now().isoformat()
        
        # Assurez-vous que l'email est correctement défini
        if 'email' not in result and 'email_templates' in result and result['email_templates']:
            # Si main.py ne fournit pas un email par défaut mais des templates, prendre le premier
            result['email'] = result['email_templates'][0]
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Résultats sauvegardés dans: {result_path}")
        
        # Stocker le nom du fichier de résultats dans la session
        session['result_filename'] = result_filename
        
        if result.get("partial_success", False):
            return jsonify({
                'success': True, 
                'partial': True,
                'message': result.get('error', 'Seules des données partielles ont pu être extraites'),
                'redirect': url_for('results')
            })
        
        return jsonify({'success': True, 'redirect': url_for('results')})
        
    except Exception as e:
        logger.error(f"Erreur dans process_cv: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})
     

@app.route('/results')
def results():
    """Affiche les résultats du traitement"""
    # Vérifier si nous avons des résultats
    result_filename = session.get('result_filename')
    if not result_filename:
        flash('Aucun résultat trouvé. Veuillez d\'abord traiter un CV.', 'warning')
        return redirect(url_for('index'))
    
    # Charger les résultats à partir du JSON sauvegardé
    result_path = os.path.join(DATA_FOLDER, result_filename)
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # Vérifier s'il s'agit d'un succès partiel
        if results.get('partial_success', False):
            flash('Seules des données partielles ont pu être extraites. Certaines informations peuvent être manquantes ou incomplètes.', 'warning')
        
        # Ajouter les variables supplémentaires pour les routes admin (résoudre l'erreur "all_candidates")
        view_only = False
        is_admin = False
        
        return render_template('results.html', results=results, view_only=view_only, is_admin=is_admin)
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des résultats: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Erreur lors du chargement des résultats: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/send_email', methods=['POST'])
def send_email():
    """Envoi de l'e-mail généré au candidat"""
    try:
        # Obtenir les données du formulaire
        recipient = request.form.get('recipient_email')
        subject = request.form.get('email_subject')
        body = request.form.get('email_body')
        
        if not all([recipient, subject, body]):
            logger.warning(f"Informations d'email manquantes: recipient={bool(recipient)}, subject={bool(subject)}, body={bool(body)}")
            return jsonify({'success': False, 'message': 'Informations d\'email manquantes'})
        
        logger.info(f"Envoi d'email à {recipient}")
        
        # Utiliser l'outil EmailSenderTool pour envoyer l'e-mail
        from main import EmailSenderTool
        email_tool = EmailSenderTool()
        email_data = {
            "recipient_email": recipient,
            "subject": subject,
            "body": body
        }
        
        result = email_tool._run(email_data)
        logger.info(f"Résultat de l'envoi d'email: {result}")
        
        # Vérifier si l'e-mail a été envoyé avec succès
        if "successfully sent" in result:
            return jsonify({'success': True, 'message': result})
        else:
            return jsonify({'success': False, 'message': result})
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'e-mail: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': f"Erreur lors de l'envoi de l'e-mail: {str(e)}"})

# Routes administratives
@app.route('/admin')
@login_required
def admin_dashboard():
    """Tableau de bord administrateur"""
    # Récupérer les descriptions de postes et les candidats
    jobs = get_all_job_descriptions()
    candidates = []
    
    try:
        for file in os.listdir(DATA_FOLDER):
            if file.endswith('_results.json'):
                file_path = os.path.join(DATA_FOLDER, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Extraire les informations clés
                        cv_data = data.get('cv_data', {})
                        job_data = data.get('job_data', {})
                        match_data = data.get('match_data', {})
                        
                        # Gérer les résultats partiels
                        is_partial = data.get('partial_success', False)
                        
                        candidate = {
                            'id': file.split('_')[0],
                            'name': cv_data.get('personal_info', {}).get('name', 'Inconnu'),
                            'job_title': job_data.get('job_title', 'Poste inconnu'),
                            'job_id': data.get('job_id', 'Inconnu'),
                            'company': job_data.get('company', 'Entreprise inconnue'),
                            'technical_match': match_data.get('technical_skills_match', 0),
                            'domain_match': match_data.get('domain_match', 0),
                            'assessment': match_data.get('final_assessment', 'Aucune évaluation'),
                            'date_processed': time.ctime(os.path.getctime(file_path)),
                            'submission_date': data.get('submission_date', 'Inconnue'),
                            'result_file': file,
                            'is_partial': is_partial
                        }
                        candidates.append(candidate)
                except Exception as e:
                    logger.error(f"Erreur lors du chargement des données du candidat depuis {file}: {str(e)}")
                    continue
                    
        # Trier par date de traitement, plus récent en premier
        candidates = sorted(candidates, key=lambda x: x.get('submission_date', ''), reverse=True)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des candidats: {str(e)}")
        flash(f'Erreur lors du chargement des candidats: {str(e)}', 'danger')
        candidates = []
        
    return render_template('admin/dashboard.html', jobs=jobs, candidates=candidates)

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Page de connexion administrateur"""
    # Si l'utilisateur est déjà connecté, rediriger vers le tableau de bord
    if session.get('logged_in'):
        return redirect(url_for('admin_dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        # Utiliser des identifiants codés en dur pour garantir l'accès
        if username == "admin" and password == "password":
            session['logged_in'] = True
            session['username'] = username
            # Assurez-vous que la session est sauvegardée
            session.modified = True
            
            flash('Connexion réussie!', 'success')
            logger.info(f"Login successful for user: {username}")
            
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            else:
                return redirect(url_for('admin_dashboard'))
        else:
            flash('Échec de la connexion. Utilisez admin/password.', 'danger')
            
    return render_template('admin/login.html')

@app.route('/admin/logout')
def admin_logout():
    """Déconnexion administrateur"""
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('Vous avez été déconnecté.', 'success')
    return redirect(url_for('admin_login'))

@app.route('/admin/jobs')
@login_required
def admin_jobs():
    """Page de gestion des postes administrateur"""
    jobs = get_all_job_descriptions()
    return render_template('admin/jobs.html', jobs=jobs)

@app.route('/admin/jobs/new', methods=['GET', 'POST'])
@login_required
def admin_new_job():
    """Créer une nouvelle description de poste"""
    if request.method == 'POST':
        try:
            # Obtenir les données du poste à partir du formulaire
            job_data = {
                'job_title': request.form.get('job_title'),
                'company': request.form.get('company'),
                'location': request.form.get('location'),
                'description': request.form.get('description'),
                'technical_skills': request.form.getlist('technical_skills'),
                'soft_skills': request.form.getlist('soft_skills'),
                'experience_years': request.form.get('experience_years'),
                'education': request.form.getlist('education'),
                'languages': request.form.getlist('languages'),
                'status': request.form.get('status', 'draft')  # draft ou published
            }
            
            # Sauvegarder la description du poste
            job_id = save_job_description(job_data)
            if job_id:
                flash('Description de poste créée avec succès!', 'success')
                return redirect(url_for('admin_jobs'))
            else:
                flash('Erreur lors de la création de la description de poste', 'danger')
        except Exception as e:
            logger.error(f"Erreur lors de la création du poste: {str(e)}")
            logger.error(traceback.format_exc())
            flash(f'Erreur lors de la création du poste: {str(e)}', 'danger')
            
    # Passer job=None explicitement pour éviter l'erreur UndefinedError
    return render_template('admin/job_form.html', job=None)

@app.route('/admin/jobs/edit/<job_id>', methods=['GET', 'POST'])
@login_required
def admin_edit_job(job_id):
    """Modifier une description de poste existante"""
    job = get_job_description(job_id)
    if not job:
        flash('Description de poste non trouvée', 'warning')
        return redirect(url_for('admin_jobs'))
    
    if request.method == 'POST':
        try:
            # Obtenir les données du poste mis à jour à partir du formulaire
            job_data = {
                'id': job_id,
                'job_title': request.form.get('job_title'),
                'company': request.form.get('company'),
                'location': request.form.get('location'),
                'description': request.form.get('description'),
                'technical_skills': request.form.getlist('technical_skills'),
                'soft_skills': request.form.getlist('soft_skills'),
                'experience_years': request.form.get('experience_years'),
                'education': request.form.getlist('education'),
                'languages': request.form.getlist('languages'),
                'status': request.form.get('status', 'draft'),  # draft ou published
                'created_at': job.get('created_at')  # Conserver la date de création
            }
            
            # Sauvegarder la description du poste mise à jour
            job_id = save_job_description(job_data)
            if job_id:
                flash('Description de poste mise à jour avec succès!', 'success')
                return redirect(url_for('admin_jobs'))
            else:
                flash('Erreur lors de la mise à jour de la description de poste', 'danger')
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du poste: {str(e)}")
            logger.error(traceback.format_exc())
            flash(f'Erreur lors de la mise à jour du poste: {str(e)}', 'danger')
            
    return render_template('admin/job_form.html', job=job)

@app.route('/admin/jobs/delete/<job_id>', methods=['POST'])
@login_required
def admin_delete_job(job_id):
    """Supprimer une description de poste"""
    if delete_job_description(job_id):
        flash('Description de poste supprimée avec succès!', 'success')
    else:
        flash('Erreur lors de la suppression de la description de poste', 'danger')
    return redirect(url_for('admin_jobs'))

@app.route('/admin/candidates')
@login_required
def admin_candidates():
    """Page de gestion des candidats administrateur"""
    candidates = []
    
    try:
        for file in os.listdir(DATA_FOLDER):
            if file.endswith('_results.json'):
                file_path = os.path.join(DATA_FOLDER, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Extraire les informations clés
                        cv_data = data.get('cv_data', {})
                        job_data = data.get('job_data', {})
                        match_data = data.get('match_data', {})
                        
                        # Gérer les résultats partiels
                        is_partial = data.get('partial_success', False)
                        
                        candidate = {
                            'id': file.split('_')[0],
                            'name': cv_data.get('personal_info', {}).get('name', 'Inconnu'),
                            'job_title': job_data.get('job_title', 'Poste inconnu'),
                            'job_id': data.get('job_id', 'Inconnu'),
                            'company': job_data.get('company', 'Entreprise inconnue'),
                            'technical_match': match_data.get('technical_skills_match', 0),
                            'domain_match': match_data.get('domain_match', 0),
                            'assessment': match_data.get('final_assessment', 'Aucune évaluation'),
                            'date_processed': time.ctime(os.path.getctime(file_path)),
                            'submission_date': data.get('submission_date', 'Inconnue'),
                            'result_file': file,
                            'is_partial': is_partial
                        }
                        candidates.append(candidate)
                except Exception as e:
                    logger.error(f"Erreur lors du chargement des données du candidat depuis {file}: {str(e)}")
                    continue
                    
        # Trier par date de traitement, plus récent en premier
        candidates = sorted(candidates, key=lambda x: x.get('submission_date', ''), reverse=True)
        
        return render_template('admin/candidates.html', candidates=candidates)
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des candidats: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Erreur lors du chargement des candidats: {str(e)}', 'danger')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/candidate/<candidate_id>')
@login_required
def admin_view_candidate(candidate_id):
    """Afficher des informations détaillées pour un candidat spécifique en tant qu'administrateur"""
    try:
        # Trouver le fichier de résultats pour ce candidat
        for file in os.listdir(DATA_FOLDER):
            if file.startswith(candidate_id) and file.endswith('_results.json'):
                file_path = os.path.join(DATA_FOLDER, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # S'assurer que l'ID est défini pour permettre la suppression
                if 'id' not in results:
                    results['id'] = candidate_id
                
                return render_template('results.html', results=results, view_only=True, is_admin=True)
        
        logger.warning(f"Candidat non trouvé: {candidate_id}")
        flash('Candidat non trouvé', 'warning')
        return redirect(url_for('admin_candidates'))
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du candidat {candidate_id}: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Erreur lors du chargement du candidat: {str(e)}', 'danger')
        return redirect(url_for('admin_candidates'))


@app.route('/admin/delete_candidate/<candidate_id>', methods=['POST'])
@login_required
def admin_delete_candidate(candidate_id):
    """Supprimer l'enregistrement d'un candidat du système"""
    try:
        # Trouver le fichier de résultats pour ce candidat
        file_to_delete = None
        for file in os.listdir(DATA_FOLDER):
            if file.startswith(candidate_id) and file.endswith('_results.json'):
                file_to_delete = os.path.join(DATA_FOLDER, file)
                break
        
        if file_to_delete and os.path.exists(file_to_delete):
            # Supprimer le fichier
            os.remove(file_to_delete)
            logger.info(f"Fichier candidat supprimé: {file_to_delete}")
            flash(f'Candidat supprimé avec succès', 'success')
        else:
            logger.warning(f"Fichier candidat non trouvé pour ID: {candidate_id}")
            flash('Fichier candidat non trouvé', 'warning')
            
        return redirect(url_for('admin_candidates'))
        
    except Exception as e:
        logger.error(f"Erreur lors de la suppression du candidat {candidate_id}: {str(e)}")
        logger.error(traceback.format_exc())
        flash(f'Erreur lors de la suppression du candidat: {str(e)}', 'danger')
        return redirect(url_for('admin_candidates'))
    
@app.errorhandler(413)
def request_entity_too_large(error):
    flash('Fichier trop volumineux. Taille maximale: 16MB.', 'danger')
    return redirect(url_for('index')), 413
    
@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Erreur interne du serveur: {str(error)}")
    logger.error(traceback.format_exc())
    return render_template('error.html', error="Erreur interne du serveur"), 500

@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html', error="Page non trouvée"), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)