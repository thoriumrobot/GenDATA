/*
    @Positive
 * Copyright (c) 2002, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.lang;

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.IndexFor;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;
    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.lock.qual.NewObject;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.checker.signedness.qual.SignednessGlb;
    @Positive
import org.checkerframework.common.value.qual.ArrayLen;
    @Positive
import org.checkerframework.common.value.qual.IntRange;
    @Positive
import org.checkerframework.common.value.qual.IntVal;
    @Positive
import org.checkerframework.common.value.qual.PolyValue;
    @Positive
import org.checkerframework.common.value.qual.StaticallyExecutable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.misc.CDS;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import java.lang.constant.Constable;
    @Positive
import java.lang.constant.DynamicConstantDesc;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Optional;
    @Positive
import static java.lang.constant.ConstantDescs.BSM_EXPLICIT_CAST;
    @Positive
import static java.lang.constant.ConstantDescs.CD_char;
    @Positive
import static java.lang.constant.ConstantDescs.CD_int;
    @Positive
import static java.lang.constant.ConstantDescs.DEFAULT_NAME;

    @Positive
@AnnotatedFor({ "index", "interning", "nullness", "value" })
    @Positive
@jdk.internal.ValueBased
    @Positive
public final class Character implements java.io.Serializable, Comparable<Character>, Constable {

    @Positive
    @Positive
    @Positive
    @IntVal(2)
    @Positive
    public static final int MIN_RADIX;

    @Positive
    @Positive
    @Positive
    @IntVal(36)
    @Positive
    public static final int MAX_RADIX;

    @Positive
    @IntVal(0)
    @Positive
    public static final char MIN_VALUE;

    @Positive
    @IntVal(65535)
    @Positive
    public static final char MAX_VALUE;

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public static final Class<Character> TYPE;

    @Positive
    @IntVal(0)
    @Positive
    public static final byte UNASSIGNED;

    @Positive
    @IntVal(1)
    @Positive
    public static final byte UPPERCASE_LETTER;

    @Positive
    @IntVal(2)
    @Positive
    public static final byte LOWERCASE_LETTER;

    @Positive
    @IntVal(3)
    @Positive
    public static final byte TITLECASE_LETTER;

    @Positive
    @IntVal(4)
    @Positive
    public static final byte MODIFIER_LETTER;

    @Positive
    @IntVal(5)
    @Positive
    public static final byte OTHER_LETTER;

    @Positive
    @IntVal(6)
    @Positive
    public static final byte NON_SPACING_MARK;

    @Positive
    @IntVal(7)
    @Positive
    public static final byte ENCLOSING_MARK;

    @Positive
    @IntVal(8)
    @Positive
    public static final byte COMBINING_SPACING_MARK;

    @Positive
    @IntVal(9)
    @Positive
    public static final byte DECIMAL_DIGIT_NUMBER;

    @Positive
    @IntVal(10)
    @Positive
    public static final byte LETTER_NUMBER;

    @Positive
    @IntVal(11)
    @Positive
    public static final byte OTHER_NUMBER;

    @Positive
    @IntVal(12)
    @Positive
    public static final byte SPACE_SEPARATOR;

    @Positive
    @IntVal(13)
    @Positive
    public static final byte LINE_SEPARATOR;

    @Positive
    @IntVal(14)
    @Positive
    public static final byte PARAGRAPH_SEPARATOR;

    @Positive
    @IntVal(15)
    @Positive
    public static final byte CONTROL;

    @Positive
    @IntVal(16)
    @Positive
    public static final byte FORMAT;

    @Positive
    @IntVal(18)
    @Positive
    public static final byte PRIVATE_USE;

    @Positive
    @IntVal(19)
    @Positive
    public static final byte SURROGATE;

    @Positive
    @IntVal(20)
    @Positive
    public static final byte DASH_PUNCTUATION;

    @Positive
    @IntVal(21)
    @Positive
    public static final byte START_PUNCTUATION;

    @Positive
    @IntVal(22)
    @Positive
    public static final byte END_PUNCTUATION;

    @Positive
    @IntVal(23)
    @Positive
    public static final byte CONNECTOR_PUNCTUATION;

    @Positive
    @IntVal(24)
    @Positive
    public static final byte OTHER_PUNCTUATION;

    @Positive
    @IntVal(25)
    @Positive
    public static final byte MATH_SYMBOL;

    @Positive
    @IntVal(26)
    @Positive
    public static final byte CURRENCY_SYMBOL;

    @Positive
    @IntVal(27)
    @Positive
    public static final byte MODIFIER_SYMBOL;

    @Positive
    @IntVal(28)
    @Positive
    public static final byte OTHER_SYMBOL;

    @Positive
    @IntVal(29)
    @Positive
    public static final byte INITIAL_QUOTE_PUNCTUATION;

    @Positive
    @IntVal(30)
    @Positive
    public static final byte FINAL_QUOTE_PUNCTUATION;

    @Positive
    @IntVal(-1)
    @Positive
    public static final byte DIRECTIONALITY_UNDEFINED;

    @Positive
    @IntVal(0)
    @Positive
    public static final byte DIRECTIONALITY_LEFT_TO_RIGHT;

    @Positive
    @IntVal(1)
    @Positive
    public static final byte DIRECTIONALITY_RIGHT_TO_LEFT;

    @Positive
    @IntVal(2)
    @Positive
    public static final byte DIRECTIONALITY_RIGHT_TO_LEFT_ARABIC;

    @Positive
    @IntVal(3)
    @Positive
    public static final byte DIRECTIONALITY_EUROPEAN_NUMBER;

    @Positive
    @IntVal(4)
    @Positive
    public static final byte DIRECTIONALITY_EUROPEAN_NUMBER_SEPARATOR;

    @Positive
    @IntVal(5)
    @Positive
    public static final byte DIRECTIONALITY_EUROPEAN_NUMBER_TERMINATOR;

    @Positive
    @IntVal(6)
    @Positive
    public static final byte DIRECTIONALITY_ARABIC_NUMBER;

    @Positive
    @IntVal(7)
    @Positive
    public static final byte DIRECTIONALITY_COMMON_NUMBER_SEPARATOR;

    @Positive
    @IntVal(8)
    @Positive
    public static final byte DIRECTIONALITY_NONSPACING_MARK;

    @Positive
    @IntVal(9)
    @Positive
    public static final byte DIRECTIONALITY_BOUNDARY_NEUTRAL;

    @Positive
    @IntVal(10)
    @Positive
    public static final byte DIRECTIONALITY_PARAGRAPH_SEPARATOR;

    @Positive
    @IntVal(11)
    @Positive
    public static final byte DIRECTIONALITY_SEGMENT_SEPARATOR;

    @Positive
    @IntVal(12)
    @Positive
    public static final byte DIRECTIONALITY_WHITESPACE;

    @Positive
    @IntVal(13)
    @Positive
    public static final byte DIRECTIONALITY_OTHER_NEUTRALS;

    @Positive
    @IntVal(14)
    @Positive
    public static final byte DIRECTIONALITY_LEFT_TO_RIGHT_EMBEDDING;

    @Positive
    @IntVal(15)
    @Positive
    public static final byte DIRECTIONALITY_LEFT_TO_RIGHT_OVERRIDE;

    @Positive
    @IntVal(16)
    @Positive
    public static final byte DIRECTIONALITY_RIGHT_TO_LEFT_EMBEDDING;

    @Positive
    @IntVal(17)
    @Positive
    public static final byte DIRECTIONALITY_RIGHT_TO_LEFT_OVERRIDE;

    @Positive
    @IntVal(18)
    @Positive
    public static final byte DIRECTIONALITY_POP_DIRECTIONAL_FORMAT;

    @Positive
    @IntVal(19)
    @Positive
    public static final byte DIRECTIONALITY_LEFT_TO_RIGHT_ISOLATE;

    @Positive
    @IntVal(20)
    @Positive
    public static final byte DIRECTIONALITY_RIGHT_TO_LEFT_ISOLATE;

    @Positive
    @IntVal(21)
    @Positive
    public static final byte DIRECTIONALITY_FIRST_STRONG_ISOLATE;

    @Positive
    @IntVal(22)
    @Positive
    public static final byte DIRECTIONALITY_POP_DIRECTIONAL_ISOLATE;

    @Positive
    @IntVal('\uD800')
    @Positive
    public static final char MIN_HIGH_SURROGATE;

    @Positive
    @IntVal('\uDBFF')
    @Positive
    public static final char MAX_HIGH_SURROGATE;

    @Positive
    @IntVal('\uDC00')
    @Positive
    public static final char MIN_LOW_SURROGATE;

    @Positive
    @IntVal('\uDFFF')
    @Positive
    public static final char MAX_LOW_SURROGATE;

    @Positive
    @IntVal(MIN_HIGH_SURROGATE)
    @Positive
    public static final char MIN_SURROGATE;

    @Positive
    @IntVal(MAX_LOW_SURROGATE)
    @Positive
    public static final char MAX_SURROGATE;

    @Positive
    @SignedPositive
    @Positive
    @IntVal(0x010000)
    @Positive
    public static final int MIN_SUPPLEMENTARY_CODE_POINT;

    @Positive
    @SignedPositive
    @Positive
    @IntVal(0x000000)
    @Positive
    public static final int MIN_CODE_POINT;

    @Positive
    @SignednessGlb
    @Positive
    @IntVal(0x000000)
    @Positive
    public static final int MAX_CODE_POINT;

    @Positive
    @Override
    @Positive
    public Optional<DynamicConstantDesc<Character>> describeConstable();

    @Positive
    public static class Subset {

    @Positive
        protected Subset(String name) {
    @Positive
        }

    @Positive
        @Pure
    @Positive
        @StaticallyExecutable
    @Positive
        public final boolean equals(@Nullable Object obj);

    @Positive
        @Pure
    @Positive
        public final int hashCode();

    @Positive
        @SideEffectFree
    @Positive
        public final String toString();
    @Positive
    }

    @Positive
    @Interned
    @Positive
    public static final class UnicodeBlock extends Subset {

    @Positive
        public static final UnicodeBlock BASIC_LATIN;

    @Positive
        public static final UnicodeBlock LATIN_1_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock LATIN_EXTENDED_A;

    @Positive
        public static final UnicodeBlock LATIN_EXTENDED_B;

    @Positive
        public static final UnicodeBlock IPA_EXTENSIONS;

    @Positive
        public static final UnicodeBlock SPACING_MODIFIER_LETTERS;

    @Positive
        public static final UnicodeBlock COMBINING_DIACRITICAL_MARKS;

    @Positive
        public static final UnicodeBlock GREEK;

    @Positive
        public static final UnicodeBlock CYRILLIC;

    @Positive
        public static final UnicodeBlock ARMENIAN;

    @Positive
        public static final UnicodeBlock HEBREW;

    @Positive
        public static final UnicodeBlock ARABIC;

    @Positive
        public static final UnicodeBlock DEVANAGARI;

    @Positive
        public static final UnicodeBlock BENGALI;

    @Positive
        public static final UnicodeBlock GURMUKHI;

    @Positive
        public static final UnicodeBlock GUJARATI;

    @Positive
        public static final UnicodeBlock ORIYA;

    @Positive
        public static final UnicodeBlock TAMIL;

    @Positive
        public static final UnicodeBlock TELUGU;

    @Positive
        public static final UnicodeBlock KANNADA;

    @Positive
        public static final UnicodeBlock MALAYALAM;

    @Positive
        public static final UnicodeBlock THAI;

    @Positive
        public static final UnicodeBlock LAO;

    @Positive
        public static final UnicodeBlock TIBETAN;

    @Positive
        public static final UnicodeBlock GEORGIAN;

    @Positive
        public static final UnicodeBlock HANGUL_JAMO;

    @Positive
        public static final UnicodeBlock LATIN_EXTENDED_ADDITIONAL;

    @Positive
        public static final UnicodeBlock GREEK_EXTENDED;

    @Positive
        public static final UnicodeBlock GENERAL_PUNCTUATION;

    @Positive
        public static final UnicodeBlock SUPERSCRIPTS_AND_SUBSCRIPTS;

    @Positive
        public static final UnicodeBlock CURRENCY_SYMBOLS;

    @Positive
        public static final UnicodeBlock COMBINING_MARKS_FOR_SYMBOLS;

    @Positive
        public static final UnicodeBlock LETTERLIKE_SYMBOLS;

    @Positive
        public static final UnicodeBlock NUMBER_FORMS;

    @Positive
        public static final UnicodeBlock ARROWS;

    @Positive
        public static final UnicodeBlock MATHEMATICAL_OPERATORS;

    @Positive
        public static final UnicodeBlock MISCELLANEOUS_TECHNICAL;

    @Positive
        public static final UnicodeBlock CONTROL_PICTURES;

    @Positive
        public static final UnicodeBlock OPTICAL_CHARACTER_RECOGNITION;

    @Positive
        public static final UnicodeBlock ENCLOSED_ALPHANUMERICS;

    @Positive
        public static final UnicodeBlock BOX_DRAWING;

    @Positive
        public static final UnicodeBlock BLOCK_ELEMENTS;

    @Positive
        public static final UnicodeBlock GEOMETRIC_SHAPES;

    @Positive
        public static final UnicodeBlock MISCELLANEOUS_SYMBOLS;

    @Positive
        public static final UnicodeBlock DINGBATS;

    @Positive
        public static final UnicodeBlock CJK_SYMBOLS_AND_PUNCTUATION;

    @Positive
        public static final UnicodeBlock HIRAGANA;

    @Positive
        public static final UnicodeBlock KATAKANA;

    @Positive
        public static final UnicodeBlock BOPOMOFO;

    @Positive
        public static final UnicodeBlock HANGUL_COMPATIBILITY_JAMO;

    @Positive
        public static final UnicodeBlock KANBUN;

    @Positive
        public static final UnicodeBlock ENCLOSED_CJK_LETTERS_AND_MONTHS;

    @Positive
        public static final UnicodeBlock CJK_COMPATIBILITY;

    @Positive
        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS;

    @Positive
        public static final UnicodeBlock HANGUL_SYLLABLES;

    @Positive
        public static final UnicodeBlock PRIVATE_USE_AREA;

    @Positive
        public static final UnicodeBlock CJK_COMPATIBILITY_IDEOGRAPHS;

    @Positive
        public static final UnicodeBlock ALPHABETIC_PRESENTATION_FORMS;

    @Positive
        public static final UnicodeBlock ARABIC_PRESENTATION_FORMS_A;

    @Positive
        public static final UnicodeBlock COMBINING_HALF_MARKS;

    @Positive
        public static final UnicodeBlock CJK_COMPATIBILITY_FORMS;

    @Positive
        public static final UnicodeBlock SMALL_FORM_VARIANTS;

    @Positive
        public static final UnicodeBlock ARABIC_PRESENTATION_FORMS_B;

    @Positive
        public static final UnicodeBlock HALFWIDTH_AND_FULLWIDTH_FORMS;

    @Positive
        public static final UnicodeBlock SPECIALS;

    @Positive
        @Deprecated()
    @Positive
        public static final UnicodeBlock SURROGATES_AREA;

    @Positive
        public static final UnicodeBlock SYRIAC;

    @Positive
        public static final UnicodeBlock THAANA;

    @Positive
        public static final UnicodeBlock SINHALA;

    @Positive
        public static final UnicodeBlock MYANMAR;

    @Positive
        public static final UnicodeBlock ETHIOPIC;

    @Positive
        public static final UnicodeBlock CHEROKEE;

    @Positive
        public static final UnicodeBlock UNIFIED_CANADIAN_ABORIGINAL_SYLLABICS;

    @Positive
        public static final UnicodeBlock OGHAM;

    @Positive
        public static final UnicodeBlock RUNIC;

    @Positive
        public static final UnicodeBlock KHMER;

    @Positive
        public static final UnicodeBlock MONGOLIAN;

    @Positive
        public static final UnicodeBlock BRAILLE_PATTERNS;

    @Positive
        public static final UnicodeBlock CJK_RADICALS_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock KANGXI_RADICALS;

    @Positive
        public static final UnicodeBlock IDEOGRAPHIC_DESCRIPTION_CHARACTERS;

    @Positive
        public static final UnicodeBlock BOPOMOFO_EXTENDED;

    @Positive
        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A;

    @Positive
        public static final UnicodeBlock YI_SYLLABLES;

    @Positive
        public static final UnicodeBlock YI_RADICALS;

    @Positive
        public static final UnicodeBlock CYRILLIC_SUPPLEMENTARY;

    @Positive
        public static final UnicodeBlock TAGALOG;

    @Positive
        public static final UnicodeBlock HANUNOO;

    @Positive
        public static final UnicodeBlock BUHID;

    @Positive
        public static final UnicodeBlock TAGBANWA;

    @Positive
        public static final UnicodeBlock LIMBU;

    @Positive
        public static final UnicodeBlock TAI_LE;

    @Positive
        public static final UnicodeBlock KHMER_SYMBOLS;

    @Positive
        public static final UnicodeBlock PHONETIC_EXTENSIONS;

    @Positive
        public static final UnicodeBlock MISCELLANEOUS_MATHEMATICAL_SYMBOLS_A;

    @Positive
        public static final UnicodeBlock SUPPLEMENTAL_ARROWS_A;

    @Positive
        public static final UnicodeBlock SUPPLEMENTAL_ARROWS_B;

    @Positive
        public static final UnicodeBlock MISCELLANEOUS_MATHEMATICAL_SYMBOLS_B;

    @Positive
        public static final UnicodeBlock SUPPLEMENTAL_MATHEMATICAL_OPERATORS;

    @Positive
        public static final UnicodeBlock MISCELLANEOUS_SYMBOLS_AND_ARROWS;

    @Positive
        public static final UnicodeBlock KATAKANA_PHONETIC_EXTENSIONS;

    @Positive
        public static final UnicodeBlock YIJING_HEXAGRAM_SYMBOLS;

    @Positive
        public static final UnicodeBlock VARIATION_SELECTORS;

    @Positive
        public static final UnicodeBlock LINEAR_B_SYLLABARY;

    @Positive
        public static final UnicodeBlock LINEAR_B_IDEOGRAMS;

    @Positive
        public static final UnicodeBlock AEGEAN_NUMBERS;

    @Positive
        public static final UnicodeBlock OLD_ITALIC;

    @Positive
        public static final UnicodeBlock GOTHIC;

    @Positive
        public static final UnicodeBlock UGARITIC;

    @Positive
        public static final UnicodeBlock DESERET;

    @Positive
        public static final UnicodeBlock SHAVIAN;

    @Positive
        public static final UnicodeBlock OSMANYA;

    @Positive
        public static final UnicodeBlock CYPRIOT_SYLLABARY;

    @Positive
        public static final UnicodeBlock BYZANTINE_MUSICAL_SYMBOLS;

    @Positive
        public static final UnicodeBlock MUSICAL_SYMBOLS;

    @Positive
        public static final UnicodeBlock TAI_XUAN_JING_SYMBOLS;

    @Positive
        public static final UnicodeBlock MATHEMATICAL_ALPHANUMERIC_SYMBOLS;

    @Positive
        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_B;

    @Positive
        public static final UnicodeBlock CJK_COMPATIBILITY_IDEOGRAPHS_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock TAGS;

    @Positive
        public static final UnicodeBlock VARIATION_SELECTORS_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock SUPPLEMENTARY_PRIVATE_USE_AREA_A;

    @Positive
        public static final UnicodeBlock SUPPLEMENTARY_PRIVATE_USE_AREA_B;

    @Positive
        public static final UnicodeBlock HIGH_SURROGATES;

    @Positive
        public static final UnicodeBlock HIGH_PRIVATE_USE_SURROGATES;

    @Positive
        public static final UnicodeBlock LOW_SURROGATES;

    @Positive
        public static final UnicodeBlock ARABIC_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock NKO;

    @Positive
        public static final UnicodeBlock SAMARITAN;

    @Positive
        public static final UnicodeBlock MANDAIC;

    @Positive
        public static final UnicodeBlock ETHIOPIC_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock UNIFIED_CANADIAN_ABORIGINAL_SYLLABICS_EXTENDED;

    @Positive
        public static final UnicodeBlock NEW_TAI_LUE;

    @Positive
        public static final UnicodeBlock BUGINESE;

    @Positive
        public static final UnicodeBlock TAI_THAM;

    @Positive
        public static final UnicodeBlock BALINESE;

    @Positive
        public static final UnicodeBlock SUNDANESE;

    @Positive
        public static final UnicodeBlock BATAK;

    @Positive
        public static final UnicodeBlock LEPCHA;

    @Positive
        public static final UnicodeBlock OL_CHIKI;

    @Positive
        public static final UnicodeBlock VEDIC_EXTENSIONS;

    @Positive
        public static final UnicodeBlock PHONETIC_EXTENSIONS_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock COMBINING_DIACRITICAL_MARKS_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock GLAGOLITIC;

    @Positive
        public static final UnicodeBlock LATIN_EXTENDED_C;

    @Positive
        public static final UnicodeBlock COPTIC;

    @Positive
        public static final UnicodeBlock GEORGIAN_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock TIFINAGH;

    @Positive
        public static final UnicodeBlock ETHIOPIC_EXTENDED;

    @Positive
        public static final UnicodeBlock CYRILLIC_EXTENDED_A;

    @Positive
        public static final UnicodeBlock SUPPLEMENTAL_PUNCTUATION;

    @Positive
        public static final UnicodeBlock CJK_STROKES;

    @Positive
        public static final UnicodeBlock LISU;

    @Positive
        public static final UnicodeBlock VAI;

    @Positive
        public static final UnicodeBlock CYRILLIC_EXTENDED_B;

    @Positive
        public static final UnicodeBlock BAMUM;

    @Positive
        public static final UnicodeBlock MODIFIER_TONE_LETTERS;

    @Positive
        public static final UnicodeBlock LATIN_EXTENDED_D;

    @Positive
        public static final UnicodeBlock SYLOTI_NAGRI;

    @Positive
        public static final UnicodeBlock COMMON_INDIC_NUMBER_FORMS;

    @Positive
        public static final UnicodeBlock PHAGS_PA;

    @Positive
        public static final UnicodeBlock SAURASHTRA;

    @Positive
        public static final UnicodeBlock DEVANAGARI_EXTENDED;

    @Positive
        public static final UnicodeBlock KAYAH_LI;

    @Positive
        public static final UnicodeBlock REJANG;

    @Positive
        public static final UnicodeBlock HANGUL_JAMO_EXTENDED_A;

    @Positive
        public static final UnicodeBlock JAVANESE;

    @Positive
        public static final UnicodeBlock CHAM;

    @Positive
        public static final UnicodeBlock MYANMAR_EXTENDED_A;

    @Positive
        public static final UnicodeBlock TAI_VIET;

    @Positive
        public static final UnicodeBlock ETHIOPIC_EXTENDED_A;

    @Positive
        public static final UnicodeBlock MEETEI_MAYEK;

    @Positive
        public static final UnicodeBlock HANGUL_JAMO_EXTENDED_B;

    @Positive
        public static final UnicodeBlock VERTICAL_FORMS;

    @Positive
        public static final UnicodeBlock ANCIENT_GREEK_NUMBERS;

    @Positive
        public static final UnicodeBlock ANCIENT_SYMBOLS;

    @Positive
        public static final UnicodeBlock PHAISTOS_DISC;

    @Positive
        public static final UnicodeBlock LYCIAN;

    @Positive
        public static final UnicodeBlock CARIAN;

    @Positive
        public static final UnicodeBlock OLD_PERSIAN;

    @Positive
        public static final UnicodeBlock IMPERIAL_ARAMAIC;

    @Positive
        public static final UnicodeBlock PHOENICIAN;

    @Positive
        public static final UnicodeBlock LYDIAN;

    @Positive
        public static final UnicodeBlock KHAROSHTHI;

    @Positive
        public static final UnicodeBlock OLD_SOUTH_ARABIAN;

    @Positive
        public static final UnicodeBlock AVESTAN;

    @Positive
        public static final UnicodeBlock INSCRIPTIONAL_PARTHIAN;

    @Positive
        public static final UnicodeBlock INSCRIPTIONAL_PAHLAVI;

    @Positive
        public static final UnicodeBlock OLD_TURKIC;

    @Positive
        public static final UnicodeBlock RUMI_NUMERAL_SYMBOLS;

    @Positive
        public static final UnicodeBlock BRAHMI;

    @Positive
        public static final UnicodeBlock KAITHI;

    @Positive
        public static final UnicodeBlock CUNEIFORM;

    @Positive
        public static final UnicodeBlock CUNEIFORM_NUMBERS_AND_PUNCTUATION;

    @Positive
        public static final UnicodeBlock EGYPTIAN_HIEROGLYPHS;

    @Positive
        public static final UnicodeBlock BAMUM_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock KANA_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock ANCIENT_GREEK_MUSICAL_NOTATION;

    @Positive
        public static final UnicodeBlock COUNTING_ROD_NUMERALS;

    @Positive
        public static final UnicodeBlock MAHJONG_TILES;

    @Positive
        public static final UnicodeBlock DOMINO_TILES;

    @Positive
        public static final UnicodeBlock PLAYING_CARDS;

    @Positive
        public static final UnicodeBlock ENCLOSED_ALPHANUMERIC_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock ENCLOSED_IDEOGRAPHIC_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock MISCELLANEOUS_SYMBOLS_AND_PICTOGRAPHS;

    @Positive
        public static final UnicodeBlock EMOTICONS;

    @Positive
        public static final UnicodeBlock TRANSPORT_AND_MAP_SYMBOLS;

    @Positive
        public static final UnicodeBlock ALCHEMICAL_SYMBOLS;

    @Positive
        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_C;

    @Positive
        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_D;

    @Positive
        public static final UnicodeBlock ARABIC_EXTENDED_A;

    @Positive
        public static final UnicodeBlock SUNDANESE_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock MEETEI_MAYEK_EXTENSIONS;

    @Positive
        public static final UnicodeBlock MEROITIC_HIEROGLYPHS;

    @Positive
        public static final UnicodeBlock MEROITIC_CURSIVE;

    @Positive
        public static final UnicodeBlock SORA_SOMPENG;

    @Positive
        public static final UnicodeBlock CHAKMA;

    @Positive
        public static final UnicodeBlock SHARADA;

    @Positive
        public static final UnicodeBlock TAKRI;

    @Positive
        public static final UnicodeBlock MIAO;

    @Positive
        public static final UnicodeBlock ARABIC_MATHEMATICAL_ALPHABETIC_SYMBOLS;

    @Positive
        public static final UnicodeBlock COMBINING_DIACRITICAL_MARKS_EXTENDED;

    @Positive
        public static final UnicodeBlock MYANMAR_EXTENDED_B;

    @Positive
        public static final UnicodeBlock LATIN_EXTENDED_E;

    @Positive
        public static final UnicodeBlock COPTIC_EPACT_NUMBERS;

    @Positive
        public static final UnicodeBlock OLD_PERMIC;

    @Positive
        public static final UnicodeBlock ELBASAN;

    @Positive
        public static final UnicodeBlock CAUCASIAN_ALBANIAN;

    @Positive
        public static final UnicodeBlock LINEAR_A;

    @Positive
        public static final UnicodeBlock PALMYRENE;

    @Positive
        public static final UnicodeBlock NABATAEAN;

    @Positive
        public static final UnicodeBlock OLD_NORTH_ARABIAN;

    @Positive
        public static final UnicodeBlock MANICHAEAN;

    @Positive
        public static final UnicodeBlock PSALTER_PAHLAVI;

    @Positive
        public static final UnicodeBlock MAHAJANI;

    @Positive
        public static final UnicodeBlock SINHALA_ARCHAIC_NUMBERS;

    @Positive
        public static final UnicodeBlock KHOJKI;

    @Positive
        public static final UnicodeBlock KHUDAWADI;

    @Positive
        public static final UnicodeBlock GRANTHA;

    @Positive
        public static final UnicodeBlock TIRHUTA;

    @Positive
        public static final UnicodeBlock SIDDHAM;

    @Positive
        public static final UnicodeBlock MODI;

    @Positive
        public static final UnicodeBlock WARANG_CITI;

    @Positive
        public static final UnicodeBlock PAU_CIN_HAU;

    @Positive
        public static final UnicodeBlock MRO;

    @Positive
        public static final UnicodeBlock BASSA_VAH;

    @Positive
        public static final UnicodeBlock PAHAWH_HMONG;

    @Positive
        public static final UnicodeBlock DUPLOYAN;

    @Positive
        public static final UnicodeBlock SHORTHAND_FORMAT_CONTROLS;

    @Positive
        public static final UnicodeBlock MENDE_KIKAKUI;

    @Positive
        public static final UnicodeBlock ORNAMENTAL_DINGBATS;

    @Positive
        public static final UnicodeBlock GEOMETRIC_SHAPES_EXTENDED;

    @Positive
        public static final UnicodeBlock SUPPLEMENTAL_ARROWS_C;

    @Positive
        public static final UnicodeBlock CHEROKEE_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock HATRAN;

    @Positive
        public static final UnicodeBlock OLD_HUNGARIAN;

    @Positive
        public static final UnicodeBlock MULTANI;

    @Positive
        public static final UnicodeBlock AHOM;

    @Positive
        public static final UnicodeBlock EARLY_DYNASTIC_CUNEIFORM;

    @Positive
        public static final UnicodeBlock ANATOLIAN_HIEROGLYPHS;

    @Positive
        public static final UnicodeBlock SUTTON_SIGNWRITING;

    @Positive
        public static final UnicodeBlock SUPPLEMENTAL_SYMBOLS_AND_PICTOGRAPHS;

    @Positive
        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_E;

    @Positive
        public static final UnicodeBlock SYRIAC_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock CYRILLIC_EXTENDED_C;

    @Positive
        public static final UnicodeBlock OSAGE;

    @Positive
        public static final UnicodeBlock NEWA;

    @Positive
        public static final UnicodeBlock MONGOLIAN_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock MARCHEN;

    @Positive
        public static final UnicodeBlock IDEOGRAPHIC_SYMBOLS_AND_PUNCTUATION;

    @Positive
        public static final UnicodeBlock TANGUT;

    @Positive
        public static final UnicodeBlock TANGUT_COMPONENTS;

    @Positive
        public static final UnicodeBlock KANA_EXTENDED_A;

    @Positive
        public static final UnicodeBlock GLAGOLITIC_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock ADLAM;

    @Positive
        public static final UnicodeBlock MASARAM_GONDI;

    @Positive
        public static final UnicodeBlock ZANABAZAR_SQUARE;

    @Positive
        public static final UnicodeBlock NUSHU;

    @Positive
        public static final UnicodeBlock SOYOMBO;

    @Positive
        public static final UnicodeBlock BHAIKSUKI;

    @Positive
        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_F;

    @Positive
        public static final UnicodeBlock GEORGIAN_EXTENDED;

    @Positive
        public static final UnicodeBlock HANIFI_ROHINGYA;

    @Positive
        public static final UnicodeBlock OLD_SOGDIAN;

    @Positive
        public static final UnicodeBlock SOGDIAN;

    @Positive
        public static final UnicodeBlock DOGRA;

    @Positive
        public static final UnicodeBlock GUNJALA_GONDI;

    @Positive
        public static final UnicodeBlock MAKASAR;

    @Positive
        public static final UnicodeBlock MEDEFAIDRIN;

    @Positive
        public static final UnicodeBlock MAYAN_NUMERALS;

    @Positive
        public static final UnicodeBlock INDIC_SIYAQ_NUMBERS;

    @Positive
        public static final UnicodeBlock CHESS_SYMBOLS;

    @Positive
        public static final UnicodeBlock ELYMAIC;

    @Positive
        public static final UnicodeBlock NANDINAGARI;

    @Positive
        public static final UnicodeBlock TAMIL_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock EGYPTIAN_HIEROGLYPH_FORMAT_CONTROLS;

    @Positive
        public static final UnicodeBlock SMALL_KANA_EXTENSION;

    @Positive
        public static final UnicodeBlock NYIAKENG_PUACHUE_HMONG;

    @Positive
        public static final UnicodeBlock WANCHO;

    @Positive
        public static final UnicodeBlock OTTOMAN_SIYAQ_NUMBERS;

    @Positive
        public static final UnicodeBlock SYMBOLS_AND_PICTOGRAPHS_EXTENDED_A;

    @Positive
        public static final UnicodeBlock YEZIDI;

    @Positive
        public static final UnicodeBlock CHORASMIAN;

    @Positive
        public static final UnicodeBlock DIVES_AKURU;

    @Positive
        public static final UnicodeBlock LISU_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock KHITAN_SMALL_SCRIPT;

    @Positive
        public static final UnicodeBlock TANGUT_SUPPLEMENT;

    @Positive
        public static final UnicodeBlock SYMBOLS_FOR_LEGACY_COMPUTING;

    @Positive
        public static final UnicodeBlock CJK_UNIFIED_IDEOGRAPHS_EXTENSION_G;

    @Positive
        @Pure
    @Positive
        @Nullable
    @Positive
        public static UnicodeBlock of(char c);

    @Positive
        @Pure
    @Positive
        @Nullable
    @Positive
        public static UnicodeBlock of(int codePoint);

    @Positive
        @Pure
    @Positive
        public static final UnicodeBlock forName(String blockName);
    @Positive
    }

    @Positive
    public static enum UnicodeScript {

    @Positive
        COMMON,
    @Positive
        LATIN,
    @Positive
        GREEK,
    @Positive
        CYRILLIC,
    @Positive
        ARMENIAN,
    @Positive
        HEBREW,
    @Positive
        ARABIC,
    @Positive
        SYRIAC,
    @Positive
        THAANA,
    @Positive
        DEVANAGARI,
    @Positive
        BENGALI,
    @Positive
        GURMUKHI,
    @Positive
        GUJARATI,
    @Positive
        ORIYA,
    @Positive
        TAMIL,
    @Positive
        TELUGU,
    @Positive
        KANNADA,
    @Positive
        MALAYALAM,
    @Positive
        SINHALA,
    @Positive
        THAI,
    @Positive
        LAO,
    @Positive
        TIBETAN,
    @Positive
        MYANMAR,
    @Positive
        GEORGIAN,
    @Positive
        HANGUL,
    @Positive
        ETHIOPIC,
    @Positive
        CHEROKEE,
    @Positive
        CANADIAN_ABORIGINAL,
    @Positive
        OGHAM,
    @Positive
        RUNIC,
    @Positive
        KHMER,
    @Positive
        MONGOLIAN,
    @Positive
        HIRAGANA,
    @Positive
        KATAKANA,
    @Positive
        BOPOMOFO,
    @Positive
        HAN,
    @Positive
        YI,
    @Positive
        OLD_ITALIC,
    @Positive
        GOTHIC,
    @Positive
        DESERET,
    @Positive
        INHERITED,
    @Positive
        TAGALOG,
    @Positive
        HANUNOO,
    @Positive
        BUHID,
    @Positive
        TAGBANWA,
    @Positive
        LIMBU,
    @Positive
        TAI_LE,
    @Positive
        LINEAR_B,
    @Positive
        UGARITIC,
    @Positive
        SHAVIAN,
    @Positive
        OSMANYA,
    @Positive
        CYPRIOT,
    @Positive
        BRAILLE,
    @Positive
        BUGINESE,
    @Positive
        COPTIC,
    @Positive
        NEW_TAI_LUE,
    @Positive
        GLAGOLITIC,
    @Positive
        TIFINAGH,
    @Positive
        SYLOTI_NAGRI,
    @Positive
        OLD_PERSIAN,
    @Positive
        KHAROSHTHI,
    @Positive
        BALINESE,
    @Positive
        CUNEIFORM,
    @Positive
        PHOENICIAN,
    @Positive
        PHAGS_PA,
    @Positive
        NKO,
    @Positive
        SUNDANESE,
    @Positive
        BATAK,
    @Positive
        LEPCHA,
    @Positive
        OL_CHIKI,
    @Positive
        VAI,
    @Positive
        SAURASHTRA,
    @Positive
        KAYAH_LI,
    @Positive
        REJANG,
    @Positive
        LYCIAN,
    @Positive
        CARIAN,
    @Positive
        LYDIAN,
    @Positive
        CHAM,
    @Positive
        TAI_THAM,
    @Positive
        TAI_VIET,
    @Positive
        AVESTAN,
    @Positive
        EGYPTIAN_HIEROGLYPHS,
    @Positive
        SAMARITAN,
    @Positive
        MANDAIC,
    @Positive
        LISU,
    @Positive
        BAMUM,
    @Positive
        JAVANESE,
    @Positive
        MEETEI_MAYEK,
    @Positive
        IMPERIAL_ARAMAIC,
    @Positive
        OLD_SOUTH_ARABIAN,
    @Positive
        INSCRIPTIONAL_PARTHIAN,
    @Positive
        INSCRIPTIONAL_PAHLAVI,
    @Positive
        OLD_TURKIC,
    @Positive
        BRAHMI,
    @Positive
        KAITHI,
    @Positive
        MEROITIC_HIEROGLYPHS,
    @Positive
        MEROITIC_CURSIVE,
    @Positive
        SORA_SOMPENG,
    @Positive
        CHAKMA,
    @Positive
        SHARADA,
    @Positive
        TAKRI,
    @Positive
        MIAO,
    @Positive
        CAUCASIAN_ALBANIAN,
    @Positive
        BASSA_VAH,
    @Positive
        DUPLOYAN,
    @Positive
        ELBASAN,
    @Positive
        GRANTHA,
    @Positive
        PAHAWH_HMONG,
    @Positive
        KHOJKI,
    @Positive
        LINEAR_A,
    @Positive
        MAHAJANI,
    @Positive
        MANICHAEAN,
    @Positive
        MENDE_KIKAKUI,
    @Positive
        MODI,
    @Positive
        MRO,
    @Positive
        OLD_NORTH_ARABIAN,
    @Positive
        NABATAEAN,
    @Positive
        PALMYRENE,
    @Positive
        PAU_CIN_HAU,
    @Positive
        OLD_PERMIC,
    @Positive
        PSALTER_PAHLAVI,
    @Positive
        SIDDHAM,
    @Positive
        KHUDAWADI,
    @Positive
        TIRHUTA,
    @Positive
        WARANG_CITI,
    @Positive
        AHOM,
    @Positive
        ANATOLIAN_HIEROGLYPHS,
    @Positive
        HATRAN,
    @Positive
        MULTANI,
    @Positive
        OLD_HUNGARIAN,
    @Positive
        SIGNWRITING,
    @Positive
        ADLAM,
    @Positive
        BHAIKSUKI,
    @Positive
        MARCHEN,
    @Positive
        NEWA,
    @Positive
        OSAGE,
    @Positive
        TANGUT,
    @Positive
        MASARAM_GONDI,
    @Positive
        NUSHU,
    @Positive
        SOYOMBO,
    @Positive
        ZANABAZAR_SQUARE,
    @Positive
        HANIFI_ROHINGYA,
    @Positive
        OLD_SOGDIAN,
    @Positive
        SOGDIAN,
    @Positive
        DOGRA,
    @Positive
        GUNJALA_GONDI,
    @Positive
        MAKASAR,
    @Positive
        MEDEFAIDRIN,
    @Positive
        ELYMAIC,
    @Positive
        NANDINAGARI,
    @Positive
        NYIAKENG_PUACHUE_HMONG,
    @Positive
        WANCHO,
    @Positive
        YEZIDI,
    @Positive
        CHORASMIAN,
    @Positive
        DIVES_AKURU,
    @Positive
        KHITAN_SMALL_SCRIPT,
    @Positive
        UNKNOWN;

    @Positive
        @Pure
    @Positive
        public static UnicodeScript of(int codePoint);

    @Positive
        @Pure
    @Positive
        public static final UnicodeScript forName(String scriptName);
    @Positive
    }

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    @PolyValue
    @Positive
    public Character(@PolyValue char value) {
    @Positive
    }

    @Positive
    private static class CharacterCache {
    @Positive
    }

    @Positive
    @IntrinsicCandidate
    @Positive
    @NewObject
    @Positive
    @PolyValue
    @Positive
    public static Character valueOf(@PolyValue char c);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @PolyValue
    @Positive
    @NonNegative
    @Positive
    public char charValue(@PolyValue Character this);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int hashCode(char value);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLen(1)
    @Positive
    public String toString();

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLen(1)
    @Positive
    public static String toString(char c);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    @ArrayLen(1)
    @Positive
    public static String toString(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isValidCodePoint(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isBmpCodePoint(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isSupplementaryCodePoint(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isHighSurrogate(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isLowSurrogate(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isSurrogate(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isSurrogatePair(char high, char low);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Positive
    @Positive
    public static int charCount(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int toCodePoint(char high, char low);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int codePointAt(CharSequence seq, @IndexFor({ "#1" }) int index);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int codePointAt(char[] a, @IndexFor({ "#1" }) int index);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int codePointAt(char[] a, @IndexFor({ "#1" }) int index, @IndexOrHigh({ "#1" }) int limit);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    static int codePointAtImpl(char[] a, int index, int limit);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int codePointBefore(CharSequence seq, @LTEqLengthOf({ "#1" }) @Positive int index);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int codePointBefore(char[] a, @LTEqLengthOf({ "#1" }) @Positive int index);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int codePointBefore(char[] a, @LTEqLengthOf({ "#1" }) @Positive int index, @IndexOrHigh({ "#1" }) int start);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    static int codePointBeforeImpl(char[] a, int index, int start);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static char highSurrogate(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static char lowSurrogate(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int toChars(int codePoint, char[] dst, @IndexFor({ "#2" }) int dstIndex);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static char[] toChars(int codePoint);

    @Positive
    static void toSurrogates(int codePoint, char[] dst, int index);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @NonNegative
    @Positive
    public static int codePointCount(CharSequence seq, @IndexOrHigh({ "#1" }) int beginIndex, @IndexOrHigh({ "#1" }) int endIndex);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @NonNegative
    @Positive
    public static int codePointCount(char[] a, @IndexOrHigh({ "#1" }) int offset, @IndexOrHigh({ "#1" }) int count);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    static int codePointCountImpl(char[] a, int offset, int count);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int offsetByCodePoints(CharSequence seq, @IndexOrHigh({ "#1" }) int index, int codePointOffset);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IndexOrHigh({ "#1" })
    @Positive
    public static int offsetByCodePoints(char[] a, @IndexOrHigh({ "#1" }) int start, @IndexOrHigh({ "#1" }) int count, @IndexOrHigh({ "#1" }) int index, int codePointOffset);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    static int offsetByCodePointsImpl(char[] a, int start, int count, int index, int codePointOffset);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isLowerCase(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isLowerCase(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isUpperCase(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isUpperCase(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isTitleCase(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isTitleCase(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isDigit(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isDigit(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isDefined(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isDefined(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isLetter(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isLetter(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isLetterOrDigit(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isLetterOrDigit(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    public static boolean isJavaLetter(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    public static boolean isJavaLetterOrDigit(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isAlphabetic(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isIdeographic(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isJavaIdentifierStart(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isJavaIdentifierStart(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isJavaIdentifierPart(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isJavaIdentifierPart(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isUnicodeIdentifierStart(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isUnicodeIdentifierStart(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isUnicodeIdentifierPart(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isUnicodeIdentifierPart(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isIdentifierIgnorable(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isIdentifierIgnorable(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static char toLowerCase(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int toLowerCase(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static char toUpperCase(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int toUpperCase(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static char toTitleCase(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int toTitleCase(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @GTENegativeOne
    @Positive
    public static int digit(char ch, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @GTENegativeOne
    @Positive
    public static int digit(int codePoint, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyValue
    @Positive
    public static int getNumericValue(@PolyValue char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int getNumericValue(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @Deprecated()
    @Positive
    public static boolean isSpace(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isSpaceChar(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isSpaceChar(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isWhitespace(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isWhitespace(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isISOControl(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isISOControl(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int getType(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int getType(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static char forDigit(int digit, @Positive @IntRange(from = 2, to = 36) int radix);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static byte getDirectionality(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static byte getDirectionality(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isMirrored(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static boolean isMirrored(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public int compareTo(Character anotherCharacter);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int compare(char x, char y);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    static int toUpperCaseEx(int codePoint);

    @Positive
    @SideEffectFree
    @Positive
    @StaticallyExecutable
    @Positive
    static char[] toUpperCaseCharArray(int codePoint);

    @Positive
    @Positive
    @Positive
    @IntVal(16)
    @Positive
    public static final int SIZE;

    @Positive
    @IntVal(2)
    @Positive
    public static final int BYTES;

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static char reverseBytes(char ch);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static String getName(int codePoint);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public static int codePointOf(String name);
    @Positive
}

// CFWR semantic augmentation - variant 1
