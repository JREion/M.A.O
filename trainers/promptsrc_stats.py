# save coop inference features, for channel importance statistics
from dassl.engine import TRAINER_REGISTRY

from .promptsrc import PromptSRC
from .promptsrc_noinitweight import PromptSRC_NOINIT


@TRAINER_REGISTRY.register()
class PromptSRCStats(PromptSRC_NOINIT):

    def model_inference(self, image):
        tokenized_prompts = self.model.tokenized_prompts
        prompts = self.model.prompt_learner()

        text_features = self.model.text_encoder(prompts, tokenized_prompts)
        image_features = self.model.image_encoder(image.type(self.model.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features, image_features
