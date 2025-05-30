<template>
  <v-dialog
    v-model="dialog_open"
    width="600px"
  >
    <v-card class="pa-3">
      <v-card-title>
        Export 3D File
      </v-card-title>
      <v-container>
        <h3>Layers</h3>
        <v-list>
          <v-list-item-group
            v-model="layer_selected"
          >
            <v-list-item
              v-for="(layer, index) in layer_items"
              :key="index"
              :value="index"
            >
              {{ layer.text }}
            </v-list-item>
          </v-list-item-group>
        </v-list>
        <h3
          v-if="has_layer_options"
        >
          Layer Options
        </h3>
        <jupyter-widget :widget="layer_layout"/>
        <v-select
          v-if="method_items.length > 1"
          label="Export method"
          :items="method_items"
          v-model="method_selected"
        />
        <h3>File Options</h3>
        <v-select
          label="Filetype"
          :items="filetype_items"
          v-model="filetype_selected"
        />
        <v-select
          v-if="show_compression"
          label="Compression method"
          :items="compression_items"
          v-model="compression_selected"
        />
        <v-row>
          <v-checkbox
            v-if="show_modelviewer"
            v-model="modelviewer"
            label="Export model-viewer HTML"
            hide-details
          />
        </v-row>
        <v-row>
          <v-checkbox
            v-if="show_modelviewer"
            :disabled="compression_selected !== 0 || !modelviewer"
            label="Include model-viewer layer controls"
            hide-details
          />
        </v-row>
        <v-row>
          <v-spacer></v-spacer>
          <v-btn class="mx-2" color="error" @click="cancel_dialog">Cancel</v-btn>
          <v-btn class="mx-2" color="success" @click="export_viewer">Export</v-btn>
        </v-row>
      </v-container>
    </v-card>
  </v-dialog>
</template>
