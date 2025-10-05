/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.font;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.Set;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
public final class NumericShaper implements java.io.Serializable {

    @Positive
    public static enum Range {

    @Positive
        EUROPEAN('\u0030', '\u0000', '\u0300'),
    @Positive
        ARABIC('\u0660', '\u0600', '\u0780'),
    @Positive
        EASTERN_ARABIC('\u06f0', '\u0600', '\u0780'),
    @Positive
        DEVANAGARI('\u0966', '\u0900', '\u0980'),
    @Positive
        BENGALI('\u09e6', '\u0980', '\u0a00'),
    @Positive
        GURMUKHI('\u0a66', '\u0a00', '\u0a80'),
    @Positive
        GUJARATI('\u0ae6', '\u0b00', '\u0b80'),
    @Positive
        ORIYA('\u0b66', '\u0b00', '\u0b80'),
    @Positive
        TAMIL('\u0be6', '\u0b80', '\u0c00'),
    @Positive
        TELUGU('\u0c66', '\u0c00', '\u0c80'),
    @Positive
        KANNADA('\u0ce6', '\u0c80', '\u0d00'),
    @Positive
        MALAYALAM('\u0d66', '\u0d00', '\u0d80'),
    @Positive
        THAI('\u0e50', '\u0e00', '\u0e80'),
    @Positive
        LAO('\u0ed0', '\u0e80', '\u0f00'),
    @Positive
        TIBETAN('\u0f20', '\u0f00', '\u1000'),
    @Positive
        MYANMAR('\u1040', '\u1000', '\u1080'),
    @Positive
        ETHIOPIC('\u1369', '\u1200', '\u1380') {

    @Positive
            @Override
    @Positive
            char getNumericBase();
    @Positive
        }
    @Positive
        ,
    @Positive
        KHMER('\u17e0', '\u1780', '\u1800'),
    @Positive
        MONGOLIAN('\u1810', '\u1800', '\u1900'),
    @Positive
        NKO('\u07c0', '\u07c0', '\u0800'),
    @Positive
        MYANMAR_SHAN('\u1090', '\u1000', '\u10a0'),
    @Positive
        LIMBU('\u1946', '\u1900', '\u1950'),
    @Positive
        NEW_TAI_LUE('\u19d0', '\u1980', '\u19e0'),
    @Positive
        BALINESE('\u1b50', '\u1b00', '\u1b80'),
    @Positive
        SUNDANESE('\u1bb0', '\u1b80', '\u1bc0'),
    @Positive
        LEPCHA('\u1c40', '\u1c00', '\u1c50'),
    @Positive
        OL_CHIKI('\u1c50', '\u1c50', '\u1c80'),
    @Positive
        VAI('\ua620', '\ua500', '\ua640'),
    @Positive
        SAURASHTRA('\ua8d0', '\ua880', '\ua8e0'),
    @Positive
        KAYAH_LI('\ua900', '\ua900', '\ua930'),
    @Positive
        CHAM('\uaa50', '\uaa00', '\uaa60'),
    @Positive
        TAI_THAM_HORA('\u1a80', '\u1a20', '\u1ab0'),
    @Positive
        TAI_THAM_THAM('\u1a90', '\u1a20', '\u1ab0'),
    @Positive
        JAVANESE('\ua9d0', '\ua980', '\ua9e0'),
    @Positive
        MEETEI_MAYEK('\uabf0', '\uabc0', '\uac00'),
    @Positive
        SINHALA('\u0de6', '\u0d80', '\u0e00'),
    @Positive
        MYANMAR_TAI_LAING('\ua9f0', '\ua9e0', '\uaa00');

    @Positive
        char getNumericBase();
    @Positive
    }

    @Positive
    public static final int EUROPEAN;

    @Positive
    public static final int ARABIC;

    @Positive
    public static final int EASTERN_ARABIC;

    @Positive
    public static final int DEVANAGARI;

    @Positive
    public static final int BENGALI;

    @Positive
    public static final int GURMUKHI;

    @Positive
    public static final int GUJARATI;

    @Positive
    public static final int ORIYA;

    @Positive
    public static final int TAMIL;

    @Positive
    public static final int TELUGU;

    @Positive
    public static final int KANNADA;

    @Positive
    public static final int MALAYALAM;

    @Positive
    public static final int THAI;

    @Positive
    public static final int LAO;

    @Positive
    public static final int TIBETAN;

    @Positive
    public static final int MYANMAR;

    @Positive
    public static final int ETHIOPIC;

    @Positive
    public static final int KHMER;

    @Positive
    public static final int MONGOLIAN;

    @Positive
    public static final int ALL_RANGES;

    @Positive
    public static NumericShaper getShaper(int singleRange);

    @Positive
    public static NumericShaper getShaper(Range singleRange);

    @Positive
    public static NumericShaper getContextualShaper(int ranges);

    @Positive
    public static NumericShaper getContextualShaper(Set<Range> ranges);

    @Positive
    public static NumericShaper getContextualShaper(int ranges, int defaultContext);

    @Positive
    public static NumericShaper getContextualShaper(Set<Range> ranges, Range defaultContext);

    @Positive
    public void shape(char[] text, int start, int count);

    @Positive
    public void shape(char[] text, int start, int count, int context);

    @Positive
    public void shape(char[] text, int start, int count, Range context);

    @Positive
    public boolean isContextual();

    @Positive
    public int getRanges();

    @Positive
    public Set<Range> getRangeSet();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
