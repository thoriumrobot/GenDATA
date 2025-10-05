/*
 * Copyright (c) 2002, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.lang;

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.checker.lock.qual.NewObject;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.SignedPositive;
import org.checkerframework.checker.signedness.qual.SignednessGlb;
import org.checkerframework.common.value.qual.ArrayLen;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.common.value.qual.IntVal;
import org.checkerframework.common.value.qual.PolyValue;
import org.checkerframework.common.value.qual.StaticallyExecutable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import jdk.internal.misc.CDS;
import jdk.internal.vm.annotation.IntrinsicCandidate;
import java.lang.constant.Constable;
import java.lang.constant.DynamicConstantDesc;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import static java.lang.constant.ConstantDescs.BSM_EXPLICIT_CAST;
import static java.lang.constant.ConstantDescs.CD_char;
import static java.lang.constant.ConstantDescs.CD_int;
import static java.lang.constant.ConstantDescs.DEFAULT_NAME;

@AnnotatedFor({ "index", "interning", "nullness", "value" })
@jdk.internal.ValueBased
public final class Character implements java.io.Serializable, Comparable<Character>, Constable {

    @Positive
    @IntVal(2)
    public static final int MIN_RADIX;

    @Positive
    @IntVal(36)
    public static final int MAX_RADIX;

    @IntVal(0)
    public static final char MIN_VALUE;

    @IntVal(65535)
    public static final char MAX_VALUE;

    @SuppressWarnings("unchecked")
    public static final Class<Character> TYPE;

    @IntVal(0)
    public static final byte UNASSIGNED;

    @IntVal(1)
    public static final byte UPPERCASE_LETTER;

    @IntVal(2)
    public static final byte LOWERCASE_LETTER;

    @IntVal(3)
    public static final byte TITLECASE_LETTER;

    @IntVal(4)
    public static final byte MODIFIER_LETTER;

    @IntVal(5)
    public static final byte OTHER_LETTER;

    @IntVal(6)
    public static final byte NON_SPACING_MARK;

    @IntVal(7)
    public static final byte ENCLOSING_MARK;

    @IntVal(8)
    public static final byte COMBINING_SPACING_MARK;

    @IntVal(9)
    public static final byte DECIMAL_DIGIT_NUMBER;

    @IntVal(10)
    public static final byte LETTER_NUMBER;

    @IntVal(11)
    public static final byte OTHER_NUMBER;

    @IntVal(12)
    public static final byte SPACE_SEPARATOR;

    @IntVal(13)
    public static final byte LINE_SEPARATOR;

    @IntVal(14)
    public static final byte PARAGRAPH_SEPARATOR;

    @IntVal(15)
    public static final byte CONTROL;

    @IntVal(16)
    public static final byte FORMAT;

    @IntVal(18)
    public static final byte PRIVATE_USE;

    @IntVal(19)
    public static final byte SURROGATE;

    @IntVal(20)
    public static final byte DASH_PUNCTUATION;

    @IntVal(21)
    public static final byte START_PUNCTUATION;

    @IntVal(22)
    public static final byte END_PUNCTUATION;

    @IntVal(23)
    public static final byte CONNECTOR_PUNCTUATION;

    @IntVal(24)
    public static final byte OTHER_PUNCTUATION;

    @IntVal(25)
    public static final byte MATH_SYMBOL;

    @IntVal(26)
    public static final byte CURRENCY_SYMBOL;

    @IntVal(27)
    public static final byte MODIFIER_SYMBOL;

    @IntVal(28)
    public static final byte OTHER_SYMBOL;

    @IntVal(29)
    public static final byte INITIAL_QUOTE_PUNCTUATION;

    @IntVal(30)
    public static final byte FINAL_QUOTE_PUNCTUATION;

    @IntVal(-1)
    public static final byte DIRECTIONALITY_UNDEFINED;

    @IntVal(0)
    public static final byte DIRECTIONALITY_LEFT_TO_RIGHT;

    @IntVal(1)
    public static final byte DIRECTIONALITY_RIGHT_TO_LEFT;

    @IntVal(2)
    public static final byte DIRECTIONALITY_RIGHT_TO_LEFT_ARABIC;

    @IntVal(3)
    public static final byte DIRECTIONALITY_EUROPEAN_NUMBER;

    @IntVal(4)
    public static final byte DIRECTIONALITY_EUROPEAN_NUMBER_SEPARATOR;

    @IntVal(5)
    public static final byte DIRECTIONALITY_EUROPEAN_NUMBER_TERMINATOR;

    @IntVal(6)
    public static final byte DIRECTIONALITY_ARABIC_NUMBER;

    @IntVal(7)
    public static final byte DIRECTIONALITY_COMMON_NUMBER_SEPARATOR;

    @IntVal(8)
    public static final byte DIRECTIONALITY_NONSPACING_MARK;

    @IntVal(9)
    public static final byte DIRECTIONALITY_BOUNDARY_NEUTRAL;

    @IntVal(10)
    public static final byte DIRECTIONALITY_PARAGRAPH_SEPARATOR;

    @IntVal(11)
    public static final byte DIRECTIONALITY_SEGMENT_SEPARATOR;

    @IntVal(12)
    public static final byte DIRECTIONALITY_WHITESPACE;

    @IntVal(13)
    public static final byte DIRECTIONALITY_OTHER_NEUTRALS;

    @IntVal(14)
    public static final byte DIRECTIONALITY_LEFT_TO_RIGHT_EMBEDDING;

    @IntVal(15)
    public static final byte DIRECTIONALITY_LEFT_TO_RIGHT_OVERRIDE;

    @IntVal(16)
    public static final byte DIRECTIONALITY_RIGHT_TO_LEFT_EMBEDDING;

    @IntVal(17)
    public static final byte DIRECTIONALITY_RIGHT_TO_LEFT_OVERRIDE;

    @IntVal(18)
    public static final byte DIRECTIONALITY_POP_DIRECTIONAL_FORMAT;

    @IntVal(19)
    public static final byte DIRECTIONALITY_LEFT_TO_RIGHT_ISOLATE;

    @IntVal(20)
    public static final byte DIRECTIONALITY_RIGHT_TO_LEFT_ISOLATE;

    @IntVal(21)
    public static final byte DIRECTIONALITY_FIRST_STRONG_ISOLATE;

    @IntVal(22)
    public static final byte DIRECTIONALITY_POP_DIRECTIONAL_ISOLATE;

    @IntVal('\uD800')
    public static final char MIN_HIGH_SURROGATE;

    @IntVal('\uDBFF')
    public static final char MAX_HIGH_SURROGATE;

    @IntVal('\uDC00')
    public static final char MIN_LOW_SURROGATE;

    @IntVal('\uDFFF')
    public static final char MAX_LOW_SURROGATE;

    @IntVal(MIN_HIGH_SURROGATE)
    public static final char MIN_SURROGATE;

    @IntVal(MAX_LOW_SURROGATE)
    public static final char MAX_SURROGATE;

    @SignedPositive
    @IntVal(0x010000)
    public static final int MIN_SUPPLEMENTARY_CODE_POINT;

    @SignedPositive
    @IntVal(0x000000)
    public static final int MIN_CODE_POINT;

    @SignednessGlb
    @IntVal(0x000000)
    public static final int MAX_CODE_POINT;

    @Override
    public Optional<DynamicConstantDesc<Character>> describeConstable();

    public static class Subset {

        protected Subset(String name) {
        }

        @Pure
        @StaticallyExecutable
        public final boolean equals(@Nullable Object obj);

        @Pure
        public final int hashCode();

        @SideEffectFree
        public final String toString();
    }

    @Interned
    public static final class UnicodeBlock extends Subset {

        public static final UnicodeBlock BASIC_LATIN;

        public static final UnicodeBlock LATIN_1_SUPPLEMENT;

        public static final UnicodeBlock LATIN_EXTENDED_A;

        public static final UnicodeBlock LATIN_EXTENDED_B;

        public static final UnicodeBlock IPA_EXTENSIONS;

        public static final UnicodeBlock SPACING_MODIFIER_LETTERS;

        public static final UnicodeBlock COMBINING_DIACRITICAL_MARKS;

        public static final UnicodeBlock GREEK;

        public static final UnicodeBlock CYRILLIC;

        public static final UnicodeBlock ARMENIAN;

        public static final UnicodeBlock HEBREW;

        public static final UnicodeBlock ARABIC;

        public static final UnicodeBlock DEVANAGARI;

        public static final UnicodeBlock BENGALI;

        public static final UnicodeBlock GURMUKHI;

        public static final UnicodeBlock GUJARATI;

        public static final UnicodeBlock ORIYA;

        public static final UnicodeBlock TAMIL;

        public static final UnicodeBlock TELUGU;

        public static final UnicodeBlock KANNADA;

        public static final UnicodeBlock MALAYALAM;

        public static final UnicodeBlock THAI;

        public static final UnicodeBlock LAO;

        public static final UnicodeBlock TIBETAN;

        public static final UnicodeBlock GEORGIAN;

        public static final UnicodeBlock HANGUL_JAMO;

        public static final UnicodeBlock LATIN_EXTENDED_ADDITIONAL;

        public static final UnicodeBlock GREEK_EXTENDED;

        public static final UnicodeBlock GENERAL_PUNCTUATION;

        public static final UnicodeBlock SUPERSCRIPTS_AND_SUBSCRIPTS;

        public static final UnicodeBlock CURRENCY_SYMBOLS;

        public static final UnicodeBlock COMBINING_MARKS_FOR_SYMBOLS;

        public static final UnicodeBlock LETTERLIKE_SYMBOLS;

        public static final UnicodeBlock NUMBER_FORMS;

        public static final UnicodeBlock ARROWS;

        public static final UnicodeBlock MATHEMATICAL_OPERATORS;

        public static final UnicodeBlock MISCELLANEOUS_TECHNICAL;

        public static final UnicodeBlock CONTROL_PICTURES;

        public static final UnicodeBlock OPTICAL_CHARACTER_RECOGNITION;

        public static final UnicodeBlock ENCLOSED_ALPHANUMERICS;

        public static final UnicodeBlock BOX_DRAWING;

        public static final UnicodeBlock BLOCK_ELEMENTS;

        public static final UnicodeBlock GEOMETRIC_SHAPES;

        public static final UnicodeBlock MISCELLANEOUS_SYMBOLS;

        public static final UnicodeBlock DINGBATS;

        public static final UnicodeBlock CJK_SYMBOLS_AND_PUNCTUATION;

        public static final UnicodeBlock HIRAGANA;

        public static final UnicodeBlock KATAKANA;

        public static final UnicodeBlock BOPOMOFO;

        public static final UnicodeBlock HANGUL_COMPATIBILITY_JAMO;

        public static final UnicodeBlock KANBUN;

        public static final UnicodeBlock ENCLOSED_CJK_LETTERS_AND_MONTHS;

        public static final UnicodeBlock CJK_COMPATIBILITY;

        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS;

        public static final UnicodeBlock HANGUL_SYLLABLES;

        public static final UnicodeBlock PRIVATE_USE_AREA;

        public static final UnicodeBlock CJK_COMPATIBILITY_IDEOGRAPHS;

        public static final UnicodeBlock ALPHABETIC_PRESENTATION_FORMS;

        public static final UnicodeBlock ARABIC_PRESENTATION_FORMS_A;

        public static final UnicodeBlock COMBINING_HALF_MARKS;

        public static final UnicodeBlock CJK_COMPATIBILITY_FORMS;

        public static final UnicodeBlock SMALL_FORM_VARIANTS;

        public static final UnicodeBlock ARABIC_PRESENTATION_FORMS_B;

        public static final UnicodeBlock HALFWIDTH_AND_FULLWIDTH_FORMS;

        public static final UnicodeBlock SPECIALS;

        @Deprecated()
        public static final UnicodeBlock SURROGATES_AREA;

        public static final UnicodeBlock SYRIAC;

        public static final UnicodeBlock THAANA;

        public static final UnicodeBlock SINHALA;

        public static final UnicodeBlock MYANMAR;

        public static final UnicodeBlock ETHIOPIC;

        public static final UnicodeBlock CHEROKEE;

        public static final UnicodeBlock UNIFIED_CANADIAN_ABORIGINAL_SYLLABICS;

        public static final UnicodeBlock OGHAM;

        public static final UnicodeBlock RUNIC;

        public static final UnicodeBlock KHMER;

        public static final UnicodeBlock MONGOLIAN;

        public static final UnicodeBlock BRAILLE_PATTERNS;

        public static final UnicodeBlock CJK_RADICALS_SUPPLEMENT;

        public static final UnicodeBlock KANGXI_RADICALS;

        public static final UnicodeBlock IDEOGRAPHIC_DESCRIPTION_CHARACTERS;

        public static final UnicodeBlock BOPOMOFO_EXTENDED;

        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A;

        public static final UnicodeBlock YI_SYLLABLES;

        public static final UnicodeBlock YI_RADICALS;

        public static final UnicodeBlock CYRILLIC_SUPPLEMENTARY;

        public static final UnicodeBlock TAGALOG;

        public static final UnicodeBlock HANUNOO;

        public static final UnicodeBlock BUHID;

        public static final UnicodeBlock TAGBANWA;

        public static final UnicodeBlock LIMBU;

        public static final UnicodeBlock TAI_LE;

        public static final UnicodeBlock KHMER_SYMBOLS;

        public static final UnicodeBlock PHONETIC_EXTENSIONS;

        public static final UnicodeBlock MISCELLANEOUS_MATHEMATICAL_SYMBOLS_A;

        public static final UnicodeBlock SUPPLEMENTAL_ARROWS_A;

        public static final UnicodeBlock SUPPLEMENTAL_ARROWS_B;

        public static final UnicodeBlock MISCELLANEOUS_MATHEMATICAL_SYMBOLS_B;

        public static final UnicodeBlock SUPPLEMENTAL_MATHEMATICAL_OPERATORS;

        public static final UnicodeBlock MISCELLANEOUS_SYMBOLS_AND_ARROWS;

        public static final UnicodeBlock KATAKANA_PHONETIC_EXTENSIONS;

        public static final UnicodeBlock YIJING_HEXAGRAM_SYMBOLS;

        public static final UnicodeBlock VARIATION_SELECTORS;

        public static final UnicodeBlock LINEAR_B_SYLLABARY;

        public static final UnicodeBlock LINEAR_B_IDEOGRAMS;

        public static final UnicodeBlock AEGEAN_NUMBERS;

        public static final UnicodeBlock OLD_ITALIC;

        public static final UnicodeBlock GOTHIC;

        public static final UnicodeBlock UGARITIC;

        public static final UnicodeBlock DESERET;

        public static final UnicodeBlock SHAVIAN;

        public static final UnicodeBlock OSMANYA;

        public static final UnicodeBlock CYPRIOT_SYLLABARY;

        public static final UnicodeBlock BYZANTINE_MUSICAL_SYMBOLS;

        public static final UnicodeBlock MUSICAL_SYMBOLS;

        public static final UnicodeBlock TAI_XUAN_JING_SYMBOLS;

        public static final UnicodeBlock MATHEMATICAL_ALPHANUMERIC_SYMBOLS;

        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_B;

        public static final UnicodeBlock CJK_COMPATIBILITY_IDEOGRAPHS_SUPPLEMENT;

        public static final UnicodeBlock TAGS;

        public static final UnicodeBlock VARIATION_SELECTORS_SUPPLEMENT;

        public static final UnicodeBlock SUPPLEMENTARY_PRIVATE_USE_AREA_A;

        public static final UnicodeBlock SUPPLEMENTARY_PRIVATE_USE_AREA_B;

        public static final UnicodeBlock HIGH_SURROGATES;

        public static final UnicodeBlock HIGH_PRIVATE_USE_SURROGATES;

        public static final UnicodeBlock LOW_SURROGATES;

        public static final UnicodeBlock ARABIC_SUPPLEMENT;

        public static final UnicodeBlock NKO;

        public static final UnicodeBlock SAMARITAN;

        public static final UnicodeBlock MANDAIC;

        public static final UnicodeBlock ETHIOPIC_SUPPLEMENT;

        public static final UnicodeBlock UNIFIED_CANADIAN_ABORIGINAL_SYLLABICS_EXTENDED;

        public static final UnicodeBlock NEW_TAI_LUE;

        public static final UnicodeBlock BUGINESE;

        public static final UnicodeBlock TAI_THAM;

        public static final UnicodeBlock BALINESE;

        public static final UnicodeBlock SUNDANESE;

        public static final UnicodeBlock BATAK;

        public static final UnicodeBlock LEPCHA;

        public static final UnicodeBlock OL_CHIKI;

        public static final UnicodeBlock VEDIC_EXTENSIONS;

        public static final UnicodeBlock PHONETIC_EXTENSIONS_SUPPLEMENT;

        public static final UnicodeBlock COMBINING_DIACRITICAL_MARKS_SUPPLEMENT;

        public static final UnicodeBlock GLAGOLITIC;

        public static final UnicodeBlock LATIN_EXTENDED_C;

        public static final UnicodeBlock COPTIC;

        public static final UnicodeBlock GEORGIAN_SUPPLEMENT;

        public static final UnicodeBlock TIFINAGH;

        public static final UnicodeBlock ETHIOPIC_EXTENDED;

        public static final UnicodeBlock CYRILLIC_EXTENDED_A;

        public static final UnicodeBlock SUPPLEMENTAL_PUNCTUATION;

        public static final UnicodeBlock CJK_STROKES;

        public static final UnicodeBlock LISU;

        public static final UnicodeBlock VAI;

        public static final UnicodeBlock CYRILLIC_EXTENDED_B;

        public static final UnicodeBlock BAMUM;

        public static final UnicodeBlock MODIFIER_TONE_LETTERS;

        public static final UnicodeBlock LATIN_EXTENDED_D;

        public static final UnicodeBlock SYLOTI_NAGRI;

        public static final UnicodeBlock COMMON_INDIC_NUMBER_FORMS;

        public static final UnicodeBlock PHAGS_PA;

        public static final UnicodeBlock SAURASHTRA;

        public static final UnicodeBlock DEVANAGARI_EXTENDED;

        public static final UnicodeBlock KAYAH_LI;

        public static final UnicodeBlock REJANG;

        public static final UnicodeBlock HANGUL_JAMO_EXTENDED_A;

        public static final UnicodeBlock JAVANESE;

        public static final UnicodeBlock CHAM;

        public static final UnicodeBlock MYANMAR_EXTENDED_A;

        public static final UnicodeBlock TAI_VIET;

        public static final UnicodeBlock ETHIOPIC_EXTENDED_A;

        public static final UnicodeBlock MEETEI_MAYEK;

        public static final UnicodeBlock HANGUL_JAMO_EXTENDED_B;

        public static final UnicodeBlock VERTICAL_FORMS;

        public static final UnicodeBlock ANCIENT_GREEK_NUMBERS;

        public static final UnicodeBlock ANCIENT_SYMBOLS;

        public static final UnicodeBlock PHAISTOS_DISC;

        public static final UnicodeBlock LYCIAN;

        public static final UnicodeBlock CARIAN;

        public static final UnicodeBlock OLD_PERSIAN;

        public static final UnicodeBlock IMPERIAL_ARAMAIC;

        public static final UnicodeBlock PHOENICIAN;

        public static final UnicodeBlock LYDIAN;

        public static final UnicodeBlock KHAROSHTHI;

        public static final UnicodeBlock OLD_SOUTH_ARABIAN;

        public static final UnicodeBlock AVESTAN;

        public static final UnicodeBlock INSCRIPTIONAL_PARTHIAN;

        public static final UnicodeBlock INSCRIPTIONAL_PAHLAVI;

        public static final UnicodeBlock OLD_TURKIC;

        public static final UnicodeBlock RUMI_NUMERAL_SYMBOLS;

        public static final UnicodeBlock BRAHMI;

        public static final UnicodeBlock KAITHI;

        public static final UnicodeBlock CUNEIFORM;

        public static final UnicodeBlock CUNEIFORM_NUMBERS_AND_PUNCTUATION;

        public static final UnicodeBlock EGYPTIAN_HIEROGLYPHS;

        public static final UnicodeBlock BAMUM_SUPPLEMENT;

        public static final UnicodeBlock KANA_SUPPLEMENT;

        public static final UnicodeBlock ANCIENT_GREEK_MUSICAL_NOTATION;

        public static final UnicodeBlock COUNTING_ROD_NUMERALS;

        public static final UnicodeBlock MAHJONG_TILES;

        public static final UnicodeBlock DOMINO_TILES;

        public static final UnicodeBlock PLAYING_CARDS;

        public static final UnicodeBlock ENCLOSED_ALPHANUMERIC_SUPPLEMENT;

        public static final UnicodeBlock ENCLOSED_IDEOGRAPHIC_SUPPLEMENT;

        public static final UnicodeBlock MISCELLANEOUS_SYMBOLS_AND_PICTOGRAPHS;

        public static final UnicodeBlock EMOTICONS;

        public static final UnicodeBlock TRANSPORT_AND_MAP_SYMBOLS;

        public static final UnicodeBlock ALCHEMICAL_SYMBOLS;

        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_C;

        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_D;

        public static final UnicodeBlock ARABIC_EXTENDED_A;

        public static final UnicodeBlock SUNDANESE_SUPPLEMENT;

        public static final UnicodeBlock MEETEI_MAYEK_EXTENSIONS;

        public static final UnicodeBlock MEROITIC_HIEROGLYPHS;

        public static final UnicodeBlock MEROITIC_CURSIVE;

        public static final UnicodeBlock SORA_SOMPENG;

        public static final UnicodeBlock CHAKMA;

        public static final UnicodeBlock SHARADA;

        public static final UnicodeBlock TAKRI;

        public static final UnicodeBlock MIAO;

        public static final UnicodeBlock ARABIC_MATHEMATICAL_ALPHABETIC_SYMBOLS;

        public static final UnicodeBlock COMBINING_DIACRITICAL_MARKS_EXTENDED;

        public static final UnicodeBlock MYANMAR_EXTENDED_B;

        public static final UnicodeBlock LATIN_EXTENDED_E;

        public static final UnicodeBlock COPTIC_EPACT_NUMBERS;

        public static final UnicodeBlock OLD_PERMIC;

        public static final UnicodeBlock ELBASAN;

        public static final UnicodeBlock CAUCASIAN_ALBANIAN;

        public static final UnicodeBlock LINEAR_A;

        public static final UnicodeBlock PALMYRENE;

        public static final UnicodeBlock NABATAEAN;

        public static final UnicodeBlock OLD_NORTH_ARABIAN;

        public static final UnicodeBlock MANICHAEAN;

        public static final UnicodeBlock PSALTER_PAHLAVI;

        public static final UnicodeBlock MAHAJANI;

        public static final UnicodeBlock SINHALA_ARCHAIC_NUMBERS;

        public static final UnicodeBlock KHOJKI;

        public static final UnicodeBlock KHUDAWADI;

        public static final UnicodeBlock GRANTHA;

        public static final UnicodeBlock TIRHUTA;

        public static final UnicodeBlock SIDDHAM;

        public static final UnicodeBlock MODI;

        public static final UnicodeBlock WARANG_CITI;

        public static final UnicodeBlock PAU_CIN_HAU;

        public static final UnicodeBlock MRO;

        public static final UnicodeBlock BASSA_VAH;

        public static final UnicodeBlock PAHAWH_HMONG;

        public static final UnicodeBlock DUPLOYAN;

        public static final UnicodeBlock SHORTHAND_FORMAT_CONTROLS;

        public static final UnicodeBlock MENDE_KIKAKUI;

        public static final UnicodeBlock ORNAMENTAL_DINGBATS;

        public static final UnicodeBlock GEOMETRIC_SHAPES_EXTENDED;

        public static final UnicodeBlock SUPPLEMENTAL_ARROWS_C;

        public static final UnicodeBlock CHEROKEE_SUPPLEMENT;

        public static final UnicodeBlock HATRAN;

        public static final UnicodeBlock OLD_HUNGARIAN;

        public static final UnicodeBlock MULTANI;

        public static final UnicodeBlock AHOM;

        public static final UnicodeBlock EARLY_DYNASTIC_CUNEIFORM;

        public static final UnicodeBlock ANATOLIAN_HIEROGLYPHS;

        public static final UnicodeBlock SUTTON_SIGNWRITING;

        public static final UnicodeBlock SUPPLEMENTAL_SYMBOLS_AND_PICTOGRAPHS;

        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_E;

        public static final UnicodeBlock SYRIAC_SUPPLEMENT;

        public static final UnicodeBlock CYRILLIC_EXTENDED_C;

        public static final UnicodeBlock OSAGE;

        public static final UnicodeBlock NEWA;

        public static final UnicodeBlock MONGOLIAN_SUPPLEMENT;

        public static final UnicodeBlock MARCHEN;

        public static final UnicodeBlock IDEOGRAPHIC_SYMBOLS_AND_PUNCTUATION;

        public static final UnicodeBlock TANGUT;

        public static final UnicodeBlock TANGUT_COMPONENTS;

        public static final UnicodeBlock KANA_EXTENDED_A;

        public static final UnicodeBlock GLAGOLITIC_SUPPLEMENT;

        public static final UnicodeBlock ADLAM;

        public static final UnicodeBlock MASARAM_GONDI;

        public static final UnicodeBlock ZANABAZAR_SQUARE;

        public static final UnicodeBlock NUSHU;

        public static final UnicodeBlock SOYOMBO;

        public static final UnicodeBlock BHAIKSUKI;

        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_F;

        public static final UnicodeBlock GEORGIAN_EXTENDED;

        public static final UnicodeBlock HANIFI_ROHINGYA;

        public static final UnicodeBlock OLD_SOGDIAN;

        public static final UnicodeBlock SOGDIAN;

        public static final UnicodeBlock DOGRA;

        public static final UnicodeBlock GUNJALA_GONDI;

        public static final UnicodeBlock MAKASAR;

        public static final UnicodeBlock MEDEFAIDRIN;

        public static final UnicodeBlock MAYAN_NUMERALS;

        public static final UnicodeBlock INDIC_SIYAQ_NUMBERS;

        public static final UnicodeBlock CHESS_SYMBOLS;

        public static final UnicodeBlock ELYMAIC;

        public static final UnicodeBlock NANDINAGARI;

        public static final UnicodeBlock TAMIL_SUPPLEMENT;

        public static final UnicodeBlock EGYPTIAN_HIEROGLYPH_FORMAT_CONTROLS;

        public static final UnicodeBlock SMALL_KANA_EXTENSION;

        public static final UnicodeBlock NYIAKENG_PUACHUE_HMONG;

        public static final UnicodeBlock WANCHO;

        public static final UnicodeBlock OTTOMAN_SIYAQ_NUMBERS;

        public static final UnicodeBlock SYMBOLS_AND_PICTOGRAPHS_EXTENDED_A;

        public static final UnicodeBlock YEZIDI;

        public static final UnicodeBlock CHORASMIAN;

        public static final UnicodeBlock DIVES_AKURU;

        public static final UnicodeBlock LISU_SUPPLEMENT;

        public static final UnicodeBlock KHITAN_SMALL_SCRIPT;

        public static final UnicodeBlock TANGUT_SUPPLEMENT;

        public static final UnicodeBlock SYMBOLS_FOR_LEGACY_COMPUTING;

        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_G;

        @Pure
        @Nullable
        public static UnicodeBlock of(char c);

        @Pure
        @Nullable
        public static UnicodeBlock of(int codePoint);

        @Pure
        public static final UnicodeBlock forName(String blockName);
    }

    public static enum UnicodeScript {

        COMMON,
        LATIN,
        GREEK,
        CYRILLIC,
        ARMENIAN,
        HEBREW,
        ARABIC,
        SYRIAC,
        THAANA,
        DEVANAGARI,
        BENGALI,
        GURMUKHI,
        GUJARATI,
        ORIYA,
        TAMIL,
        TELUGU,
        KANNADA,
        MALAYALAM,
        SINHALA,
        THAI,
        LAO,
        TIBETAN,
        MYANMAR,
        GEORGIAN,
        HANGUL,
        ETHIOPIC,
        CHEROKEE,
        CANADIAN_ABORIGINAL,
        OGHAM,
        RUNIC,
        KHMER,
        MONGOLIAN,
        HIRAGANA,
        KATAKANA,
        BOPOMOFO,
        HAN,
        YI,
        OLD_ITALIC,
        GOTHIC,
        DESERET,
        INHERITED,
        TAGALOG,
        HANUNOO,
        BUHID,
        TAGBANWA,
        LIMBU,
        TAI_LE,
        LINEAR_B,
        UGARITIC,
        SHAVIAN,
        OSMANYA,
        CYPRIOT,
        BRAILLE,
        BUGINESE,
        COPTIC,
        NEW_TAI_LUE,
        GLAGOLITIC,
        TIFINAGH,
        SYLOTI_NAGRI,
        OLD_PERSIAN,
        KHAROSHTHI,
        BALINESE,
        CUNEIFORM,
        PHOENICIAN,
        PHAGS_PA,
        NKO,
        SUNDANESE,
        BATAK,
        LEPCHA,
        OL_CHIKI,
        VAI,
        SAURASHTRA,
        KAYAH_LI,
        REJANG,
        LYCIAN,
        CARIAN,
        LYDIAN,
        CHAM,
        TAI_THAM,
        TAI_VIET,
        AVESTAN,
        EGYPTIAN_HIEROGLYPHS,
        SAMARITAN,
        MANDAIC,
        LISU,
        BAMUM,
        JAVANESE,
        MEETEI_MAYEK,
        IMPERIAL_ARAMAIC,
        OLD_SOUTH_ARABIAN,
        INSCRIPTIONAL_PARTHIAN,
        INSCRIPTIONAL_PAHLAVI,
        OLD_TURKIC,
        BRAHMI,
        KAITHI,
        MEROITIC_HIEROGLYPHS,
        MEROITIC_CURSIVE,
        SORA_SOMPENG,
        CHAKMA,
        SHARADA,
        TAKRI,
        MIAO,
        CAUCASIAN_ALBANIAN,
        BASSA_VAH,
        DUPLOYAN,
        ELBASAN,
        GRANTHA,
        PAHAWH_HMONG,
        KHOJKI,
        LINEAR_A,
        MAHAJANI,
        MANICHAEAN,
        MENDE_KIKAKUI,
        MODI,
        MRO,
        OLD_NORTH_ARABIAN,
        NABATAEAN,
        PALMYRENE,
        PAU_CIN_HAU,
        OLD_PERMIC,
        PSALTER_PAHLAVI,
        SIDDHAM,
        KHUDAWADI,
        TIRHUTA,
        WARANG_CITI,
        AHOM,
        ANATOLIAN_HIEROGLYPHS,
        HATRAN,
        MULTANI,
        OLD_HUNGARIAN,
        SIGNWRITING,
        ADLAM,
        BHAIKSUKI,
        MARCHEN,
        NEWA,
        OSAGE,
        TANGUT,
        MASARAM_GONDI,
        NUSHU,
        SOYOMBO,
        ZANABAZAR_SQUARE,
        HANIFI_ROHINGYA,
        OLD_SOGDIAN,
        SOGDIAN,
        DOGRA,
        GUNJALA_GONDI,
        MAKASAR,
        MEDEFAIDRIN,
        ELYMAIC,
        NANDINAGARI,
        NYIAKENG_PUACHUE_HMONG,
        WANCHO,
        YEZIDI,
        CHORASMIAN,
        DIVES_AKURU,
        KHITAN_SMALL_SCRIPT,
        UNKNOWN;

        @Pure
        public static UnicodeScript of(int codePoint);

        @Pure
        public static final UnicodeScript forName(String scriptName);
    }

    @Pure
    @StaticallyExecutable
    @Deprecated()
    @PolyValue
    public Character(@PolyValue char value) {
    }

    private static class CharacterCache {
    }

    @IntrinsicCandidate
    @NewObject
    @PolyValue
    public static Character valueOf(@PolyValue char c);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @PolyValue
    @NonNegative
    public char charValue(@PolyValue Character this);

    @Pure
    @StaticallyExecutable
    @Override
    public int hashCode();

    @Pure
    @StaticallyExecutable
    public static int hashCode(char value);

    @Pure
    @StaticallyExecutable
    public boolean equals(@Nullable Object obj);

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLen(1)
    public String toString();

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLen(1)
    public static String toString(char c);

    @SideEffectFree
    @StaticallyExecutable
    @ArrayLen(1)
    public static String toString(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isValidCodePoint(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isBmpCodePoint(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isSupplementaryCodePoint(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isHighSurrogate(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isLowSurrogate(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isSurrogate(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isSurrogatePair(char high, char low);

    @Pure
    @StaticallyExecutable
    @Positive
    public static int charCount(int codePoint);

    @Pure
    @StaticallyExecutable
    public static int toCodePoint(char high, char low);

    @Pure
    @StaticallyExecutable
    public static int codePointAt(CharSequence seq, @IndexFor({ "#1" }) int index);

    @Pure
    @StaticallyExecutable
    public static int codePointAt(char[] a, @IndexFor({ "#1" }) int index);

    @Pure
    @StaticallyExecutable
    public static int codePointAt(char[] a, @IndexFor({ "#1" }) int index, @IndexOrHigh({ "#1" }) int limit);

    @Pure
    @StaticallyExecutable
    static int codePointAtImpl(char[] a, int index, int limit);

    @Pure
    @StaticallyExecutable
    public static int codePointBefore(CharSequence seq, @LTEqLengthOf({ "#1" }) @Positive int index);

    @Pure
    @StaticallyExecutable
    public static int codePointBefore(char[] a, @LTEqLengthOf({ "#1" }) @Positive int index);

    @Pure
    @StaticallyExecutable
    public static int codePointBefore(char[] a, @LTEqLengthOf({ "#1" }) @Positive int index, @IndexOrHigh({ "#1" }) int start);

    @Pure
    @StaticallyExecutable
    static int codePointBeforeImpl(char[] a, int index, int start);

    @Pure
    @StaticallyExecutable
    public static char highSurrogate(int codePoint);

    @Pure
    @StaticallyExecutable
    public static char lowSurrogate(int codePoint);

    @Pure
    @StaticallyExecutable
    public static int toChars(int codePoint, char[] dst, @IndexFor({ "#2" }) int dstIndex);

    @Pure
    @StaticallyExecutable
    public static char[] toChars(int codePoint);

    static void toSurrogates(int codePoint, char[] dst, int index);

    @Pure
    @StaticallyExecutable
    @NonNegative
    public static int codePointCount(CharSequence seq, @IndexOrHigh({ "#1" }) int beginIndex, @IndexOrHigh({ "#1" }) int endIndex);

    @Pure
    @StaticallyExecutable
    @NonNegative
    public static int codePointCount(char[] a, @IndexOrHigh({ "#1" }) int offset, @IndexOrHigh({ "#1" }) int count);

    @Pure
    @StaticallyExecutable
    static int codePointCountImpl(char[] a, int offset, int count);

    @Pure
    @StaticallyExecutable
    public static int offsetByCodePoints(CharSequence seq, @IndexOrHigh({ "#1" }) int index, int codePointOffset);

    @Pure
    @StaticallyExecutable
    @IndexOrHigh({ "#1" })
    public static int offsetByCodePoints(char[] a, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int count, @IndexOrHigh({ "#1" }) int index, int codePointOffset);

    @Pure
    @StaticallyExecutable
    static int offsetByCodePointsImpl(char[] a, int start, int count, int index, int codePointOffset);

    @Pure
    @StaticallyExecutable
    public static boolean isLowerCase(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isLowerCase(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isUpperCase(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isUpperCase(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isTitleCase(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isTitleCase(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isDigit(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isDigit(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isDefined(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isDefined(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isLetter(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isLetter(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isLetterOrDigit(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isLetterOrDigit(int codePoint);

    @Pure
    @StaticallyExecutable
    @Deprecated()
    public static boolean isJavaLetter(char ch);

    @Pure
    @StaticallyExecutable
    @Deprecated()
    public static boolean isJavaLetterOrDigit(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isAlphabetic(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isIdeographic(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isJavaIdentifierStart(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isJavaIdentifierStart(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isJavaIdentifierPart(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isJavaIdentifierPart(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isUnicodeIdentifierStart(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isUnicodeIdentifierStart(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isUnicodeIdentifierPart(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isUnicodeIdentifierPart(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isIdentifierIgnorable(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isIdentifierIgnorable(int codePoint);

    @Pure
    @StaticallyExecutable
    public static char toLowerCase(char ch);

    @Pure
    @StaticallyExecutable
    public static int toLowerCase(int codePoint);

    @Pure
    @StaticallyExecutable
    public static char toUpperCase(char ch);

    @Pure
    @StaticallyExecutable
    public static int toUpperCase(int codePoint);

    @Pure
    @StaticallyExecutable
    public static char toTitleCase(char ch);

    @Pure
    @StaticallyExecutable
    public static int toTitleCase(int codePoint);

    @Pure
    @StaticallyExecutable
    @GTENegativeOne
    public static int digit(char ch, @Positive @IntRange(from = 2, to = 36) int radix);

    @Pure
    @StaticallyExecutable
    @GTENegativeOne
    public static int digit(int codePoint, @Positive @IntRange(from = 2, to = 36) int radix);

    @Pure
    @StaticallyExecutable
    @PolyValue
    public static int getNumericValue(@PolyValue char ch);

    @Pure
    @StaticallyExecutable
    public static int getNumericValue(int codePoint);

    @Pure
    @StaticallyExecutable
    @Deprecated()
    public static boolean isSpace(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isSpaceChar(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isSpaceChar(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isWhitespace(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isWhitespace(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isISOControl(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isISOControl(int codePoint);

    @Pure
    @StaticallyExecutable
    public static int getType(char ch);

    @Pure
    @StaticallyExecutable
    public static int getType(int codePoint);

    @Pure
    @StaticallyExecutable
    public static char forDigit(int digit, @Positive @IntRange(from = 2, to = 36) int radix);

    @Pure
    @StaticallyExecutable
    public static byte getDirectionality(char ch);

    @Pure
    @StaticallyExecutable
    public static byte getDirectionality(int codePoint);

    @Pure
    @StaticallyExecutable
    public static boolean isMirrored(char ch);

    @Pure
    @StaticallyExecutable
    public static boolean isMirrored(int codePoint);

    @Pure
    @StaticallyExecutable
    public int compareTo(Character anotherCharacter);

    @Pure
    @StaticallyExecutable
    public static int compare(char x, char y);

    @Pure
    @StaticallyExecutable
    static int toUpperCaseEx(int codePoint);

    @SideEffectFree
    @StaticallyExecutable
    static char[] toUpperCaseCharArray(int codePoint);

    @Positive
    @IntVal(16)
    public static final int SIZE;

    @IntVal(2)
    public static final int BYTES;

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static char reverseBytes(char ch);

    @Pure
    @StaticallyExecutable
    public static String getName(int codePoint);

    @Pure
    @StaticallyExecutable
    public static int codePointOf(String name);
}
