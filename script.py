
#Econtramos todos los textos problematicos en el archivos para su posterior parseo
def find_special_characters(data_path):
    with open(data_path, 'r', encoding='utf-8') as archivo:
        contenido = archivo.read()

    info_corructa = []

    for info in contenido.split(' '):
        if '�' in info:
            info_corructa.append(info)
        
    info_corructa = set(info_corructa)
    info_corructa = list(info_corructa)

    print(info_corructa)
    


# Guardamos todos los textos problematicos en un diccionario 
replacements = {
    'T�rmico': 'Térmico',
    'Caracter�sticas': 'Características',
    'S�': 'Sí',
    'F�cil': 'Fácil',
    'Instalaci�n': 'Instalación',
    'Aislamiento Ac�stico': 'Aislamiento Acústico',
    'Compensaci�n': 'Compensación',
    'Peque�as': 'Pequeñas',
    'R�pida': 'Rápida',
    'Tr�fico': 'Tráfico',
    'Pel�cula': 'Película',
    'T�cnico': 'Técnico',
    'Espa�a': 'España',
    'A�os': 'Años',
    'C�lido': 'Cálido',
    'M�rmoles': 'Mármoles',
    'S�lo': 'Sólo',
    'D�ficil': 'Difícil',
    'M�ltiples': 'Múltiples',
    't�rmico': 'térmico',
    'caracter�sticas': 'características',
    's�': 'sí',
    'f�cil': 'fácil',
    'instalaci�n': 'instalación',
    'aislamiento ac�stico': 'aislamiento acústico',
    'compensaci�n': 'compensación',
    'peque�as': 'pequeñas',
    'r�pida': 'rápida',
    'tr�fico': 'tráfico',
    'pel�cula': 'película',
    't�cnico': 'técnico',
    'espa�a': 'españa',
    'a�os': 'años',
    'c�lido': 'cálido',
    'm�rmoles': 'mármoles',
    'baldosas': 'baldosas',
    's�lo': 'sólo',
    'd�ficil': 'difícil',
    'm�ltiples': 'múltiples',
    "Protecci�n": "Protección",
    "Compensaci�n": "Compensación",
    "l�tex": "látex",
    "h�medas": "húmedas",
    "calefacci�n": "calefacción",
    "el�ctrica": "eléctrica",
    "peque�os": "pequeños",
    "ac�stica": "acústica",
    "absorci�n": "absorción",
    "Rodapi�": "Rodapié",
    "marr�n": "marrón",
    "marr�n oscuro": "marrón oscuro",
    "marr�n claro": "marrón claro",
    "certificaci�n": "certificación",
    "coordinaci�n": "coordinación",
    "gesti�n": "gestión",
    "v�lido": "válido",
    "instalaci�n": "instalación",
    "dise�ado": "diseñado",
    "ac�ido": "ácido",
        'garant�a': 'garantía',
    'Resistente': 'Resistente',  # Sin cambios
    't�rmico,': 'térmico,',
    'Protecci�n': 'Protección',
    'medici�n': 'medición',
    'm�rmoles.': 'mármoles.',
    'instalaci�n': 'instalación',
    'Mamperl�n': 'Mampérlan',
    'fijaci�n': 'fijación',
    '�/m�': '€/m²',
    'Compatible': 'Compatible',  # Sin cambios
    'mm': 'mm',  # Sin cambios
    'Marr�n': 'Marrón',
    'Rodapi�': 'Rodapié',
    'Ac�stico': 'Acústico',
    'Protecci�n': 'Protección',
    'gesti�n': 'gestión',
    'cu�as': 'cuñas',
    'cl�sico': 'clásico',
    'No': 'No',  # Sin cambios
    'ac�stico.': 'acústico.',
    'Base': 'Base',  # Sin cambios
    'r�stico,': 'rústico,',
    'c�lido,': 'cálido,',
    'Resistente': 'Resistente',  # Sin cambios
    'despu�s': 'después',
    '�ltima': 'última',
    'absorci�n': 'absorción',
    'imitaci�n': 'imitación',
    'cm': 'cm',  # Sin cambios
    'Pl�stico': 'Plástico',
    '4': '4',  # Sin cambios
    'f�cilmente': 'fácilmente',
    'marr�n': 'marrón',
    'vin�licos.': 'vinílicos.',
    'contempor�neo,': 'contemporáneo,',
    'cl�sico,': 'clásico,',
    '�/m�': '€/m²',
    'Apta': 'Apta',  # Sin cambios
    '10m�': '10m²',
    'Espuma': 'Espuma',  # Sin cambios
    'Espa�a;': 'España;',
    'rodapi�': 'rodapié',
    '�cidos.': 'ácidos.',
    '�/m�': '€/m²',
    'Proporciona': 'Proporciona',  # Sin cambios
    'decoraci�n': 'decoración',
    'pl�stico': 'plástico',
    'Banda': 'Banda',  # Sin cambios
    'Met�lico': 'Metálico',
    'Limpiador': 'Limpiador',  # Sin cambios
    'dise�ado': 'diseñado',
    '�cido': 'ácido',
    'manteni�ndola': 'manteniéndola',
    'm�': 'm²',
    '2,05': '2,05',  # Sin cambios
    '""Colocaci�n': '"Colocación',
    'el�ctrica,': 'eléctrica,',
    'cu�a:': 'cuña:',
    '�/m�': '€/m²',
    'Caja': 'Caja',  # Sin cambios
    'Antihumedad': 'Antihumedad',  # Sin cambios
    'Compensaci�n': 'Compensación',
    '10m�': '10m²',
    'Poliestireno': 'Poliestireno',
    '0,47': '0,47',  # Sin cambios
    '5,5m�': '5,5m²',
    'Espuma': 'Espuma',  # Sin cambios
    '�': 'Herramienta',
    'Abrillantador': 'Abrillantador',  # Sin cambios
    'WOLFCRAFT': 'WOLFCRAFT',  # Sin cambios
    'Pl�stico': 'Plástico',
    'ac�stica,': 'acústica,',
    'tr�fico': 'tráfico',
    'l�tex': 'látex',
    '1': '1',  # Sin cambios
    '�/m�': '€/m²',
    'Maximiza': 'Maximiza',  # Sin cambios
    'T�rmico': 'Térmico',
    'Aislamiento': 'Aislamiento',  # Sin cambios
    'Espa�a,': 'España,',
    '�': '"Acabado',
    'r�pida': 'rápida',
    'pelda�os,': 'peldaños,',
    'instalaci�n,': 'instalación,',
    'cer�mica;': 'cerámica;',
    'h�medas,': 'húmedas,',
    '�': '"Adecuado',
    '10m�': '10m²',
    'Corcho': 'Corcho',  # Sin cambios
    '0,16': '0,16',  # Sin cambios
    'm�': 'm²',
    '5,99': '5,99',  # Sin cambios
    '�/m�': '€/m²',
    'F�cil': 'Fácil',
    'garant�a.': 'garantía.',  
    'Perfil': 'Perfil',  # Sin cambios
    '�/m�': '€/m²',
    'Dise�ada': 'Diseñada',
    'tracci�n.': 'tracción.',
    'Kit': 'Kit',  # Sin cambios
    'peque�os': 'pequeños',
    'pel�cula': 'película',
    '12m�': '12m²',
    'Espuma': 'Espuma',  # Sin cambios
    '�': '"Fácil',
    'rodapi�s"""': 'rodapiés"""',
    'm�': 'm²',
    '3,19': '3,19',  # Sin cambios
    'rodapi�s.': 'rodapiés.',
    'Cu�a': 'Cuña',
    'aplica': 'aplica',  # Sin cambios
    'No': 'No',  # Sin cambios
    'S�': 'Sí',
    'S�': 'Sí',
    'S�': 'Sí',
    'Hasta': 'Hasta',  # Sin cambios
    'calc�reos,': 'calcáreos,',
    'colocaci�n': 'colocación',
    '�': 'Potente',
    'f�cil': 'fácil',
    'cm': 'cm',  # Sin cambios
    'Marr�n': 'Marrón',
    'No': 'No',  # Sin cambios
    '�/m�': '€/m²',
    'Incluye': 'Incluye',  # Sin cambios
    '�': 'Certificado',
    'cl�sico,': 'clásico,',
    'multifunci�n;': 'multifunción;',
    '�': 'Coordinado',
    'aplica': 'aplica',  # Sin cambios
    'No': 'No',  # Sin cambios
    'S�': 'Sí',
    'S�': 'Sí',
    'No': 'No',  # Sin cambios
    'Hasta': 'Hasta',  # Sin cambios
    '�': 'Detergente',
    'pol�meros': 'polímeros',
    'm�': 'm²',
    '2,39': '2,39',  # Sin cambios
    '�': '"Ideal',
    'm�': 'm²',
    '4,99': '4,99',  # Sin cambios
    '�': 'Cantos',
    'cer�micos,': 'cerámicos,',
    '�': 'Acabado',
    'Cu�as:': 'Cuñas:',
    '�': 'Pack',
    'calefacci�n': 'calefacción',
    '5m�': '5m²',
    'v�lido': 'válido',
    'sostenible': 'sostenible',
    'Mamperl�n': 'Mampérlan',
    '�': 'Kit',
    'Mamperl�n': 'Mampérlan',
    'cm': 'cm',  # Sin cambios
    'Marr�n': 'Marrón',
    'mar�timo': 'marítimo',
    '1,4': '1,4',  # Sin cambios
    'instalaci�n;': 'instalación;',
    '�/m�': '€/m²',
    'Ligera,': 'Ligera,',
    'transici�n': 'transición',
    'tamb�': 'también',
    'aplica': 'aplica',  # Sin cambios
    'S�': 'Sí',
    'S�': 'Sí',
    'S�': 'Sí',
    'Hasta': 'Hasta',  # Sin cambios
    'instalaci�n."': 'instalación."',
    'Perfil': 'Perfil',  # Sin cambios
    '�/m�': '€/m²',
    'Hecho': 'Hecho',  # Sin cambios
    'm�': 'm²',
    '4,79': '4,79',  # Sin cambios
    'concentraci�n': 'concentración',
    '48m�': '48m²',
    'Polipropileno': 'Polipropileno',
    '1,2': '1,2',  # Sin cambios
    'a�os': 'años',
    'garant�a': 'garantía',
    'Mamperl�n': 'Mampérlan',
    'f�cil.': 'fácil.',  
    'Base': 'Base',  # Sin cambios
    'Ajustador': 'Ajustador',  # Sin cambios
    'intensivo.': 'intensivo.',
    'Rodapi�': 'Rodapié',
    'instalaci�n."': 'instalación."',
    'Junquillo': 'Junquillo',  # Sin cambios
    'V�lido': 'Válido',
    'instalaci�n."': 'instalación."',
    'Kit': 'Kit',  # Sin cambios
    'rodapi�s': 'rodapiés'
}

# Leemos el archivo nuevamente y posteriormente guardamos en una nuevo toda la informacion esta vez con los datos en un formato correcto
def correct_information(data_path,replacements):
    with open(data_path, 'r', encoding='utf-8', errors='replace') as file:
        contenido = file.read()

    # Reemplazar caracteres problematicos
    for original, replacement in replacements.items():
        contenido = contenido.replace(original, replacement)

    # Guardar el contenido corregido en un nuevo archivo
    with open('data/BD_Suelos2.txt', 'w', encoding='utf-8') as file:
        file.write(contenido)

    print("El archivo ha sido corregido y guardado como data/BD_Suelos2.txt.")


    # Finalmente concatenamos ambas bases de datos
def concat_db(entrada_1,entrada_2,salida):

    with open(entrada_1, 'r', encoding='utf-8') as archivo1, \
         open(entrada_2, 'r', encoding='utf-8') as archivo2:
        contenido_1 = archivo1.read()
        contenido_2 = archivo2.read()

    with open(salida, 'w', encoding='utf-8') as archivo_salida:
        archivo_salida.write(contenido_1)
        archivo_salida.write('\n\n')  # Dos saltos de línea
        archivo_salida.write(contenido_2)


