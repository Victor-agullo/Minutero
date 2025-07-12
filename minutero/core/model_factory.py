# core/model_factory.py

import importlib.util, sys, os
from typing import Dict, Type
from core.models import BaseModel
from utils.logger import logger
from config.settings import settings

class ModelFactory:
    """
    Factory para cargar dinámicamente modelos de transcripción desde el directorio de plugins.
    """
    _registered_models: Dict[str, Type[BaseModel]] = {}

    def __init__(self):
        self._load_plugins()

    def _load_plugins(self):
        """Carga todos los módulos de plugins del directorio especificado."""
        plugins_dir = settings.PLUGINS_DIR
        logger.info(f"Buscando plugins en: {plugins_dir}")

        if not os.path.exists(plugins_dir):
            logger.warning(f"Directorio de plugins no encontrado: {plugins_dir}")
            return

        for filename in os.listdir(plugins_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3] # Eliminar .py
                file_path = os.path.join(plugins_dir, filename)
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None:
                        logger.warning(f"No se pudo cargar la especificación para el módulo {module_name} en {file_path}")
                        continue
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    logger.info(f"Módulo de plugin '{module_name}' cargado desde '{file_path}'")

                    # Buscar subclases de BaseModel dentro del módulo
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel:
                            self.register_model(module_name, attr)
                            logger.info(f"Modelo '{attr.__name__}' registrado para el plugin '{module_name}'.")

                except Exception as e:
                    logger.error(f"Error al cargar el plugin '{filename}': {e}", exc_info=True)

    def register_model(self, name: str, model_class: Type[BaseModel]):
        """
        Registra una clase de modelo con un nombre específico.
        """
        if name in self._registered_models:
            logger.warning(f"El modelo '{name}' ya está registrado. Sobrescribiendo.")
        self._registered_models[name] = model_class
        logger.info(f"Modelo '{name}' registrado.")

    def get_model_instance(self, model_name_key: str, **kwargs) -> BaseModel:
        """
        Crea una instancia de un modelo de transcripción registrado.
        model_name_key: El nombre clave del modelo (ej: 'whisper_model').
        """
        model_class = self._registered_models.get(model_name_key)
        if not model_class:
            raise ValueError(f"Modelo '{model_name_key}' no encontrado. Modelos disponibles: {list(self._registered_models.keys())}")
        return model_class(**kwargs)

    def get_available_models(self) -> list[str]:
        """Devuelve una lista de los nombres de los modelos disponibles."""
        return list(self._registered_models.keys())

# Instancia global del factory
model_factory = ModelFactory()