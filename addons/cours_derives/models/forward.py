import math
from odoo import api, fields, models

class ForwardContract(models.Model):
    _name = "cours.forward"
    _description = "Contrat forward"

    name = fields.Char(string="Reference", required=True)
    S0 = fields.Float(string="Prix spot initial", required=True)
    r = fields.Float(string="Taux sans risque (decimal)", required=True)
    T = fields.Float(string="Maturite (ans)", required=True)

    F0 = fields.Float(
        string="Prix forward initial",
        digits=(16, 4),
        compute="_compute_F0",
        store=True,
        readonly=False
    )

    t_elapsed = fields.Float(string="Temps ecoule (t)", default=0.0)
    S_t = fields.Float(string="Prix spot courant")

    value = fields.Float(
        string="Valeur du contrat",
        compute="_compute_value",
        store=True
    )

    @api.depends("S0", "r", "T")
    def _compute_F0(self):
        for rec in self:
            rec.F0 = rec.S0 * math.exp(rec.r * rec.T)

    @api.depends("S_t", "F0", "r", "T", "t_elapsed")
    def _compute_value(self):
        for rec in self:
            rec.value = (
                rec.S_t
                - rec.F0 * math.exp(-rec.r * (rec.T - rec.t_elapsed))
            )
