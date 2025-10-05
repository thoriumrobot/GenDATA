/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.font;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.util.Arrays;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.Set;
import jdk.internal.access.SharedSecrets;

public final class NumericShaper implements java.io.Serializable {

    public static enum Range {

        EUROPEAN('\u0030', '\u0000', '\u0300'),
        ARABIC('\u0660', '\u0600', '\u0780'),
        EASTERN_ARABIC('\u06f0', '\u0600', '\u0780'),
        DEVANAGARI('\u0966', '\u0900', '\u0980'),
        BENGALI('\u09e6', '\u0980', '\u0a00'),
        GURMUKHI('\u0a66', '\u0a00', '\u0a80'),
        GUJARATI('\u0ae6', '\u0b00', '\u0b80'),
        ORIYA('\u0b66', '\u0b00', '\u0b80'),
        TAMIL('\u0be6', '\u0b80', '\u0c00'),
        TELUGU('\u0c66', '\u0c00', '\u0c80'),
        KANNADA('\u0ce6', '\u0c80', '\u0d00'),
        MALAYALAM('\u0d66', '\u0d00', '\u0d80'),
        THAI('\u0e50', '\u0e00', '\u0e80'),
        LAO('\u0ed0', '\u0e80', '\u0f00'),
        TIBETAN('\u0f20', '\u0f00', '\u1000'),
        MYANMAR('\u1040', '\u1000', '\u1080'),
        ETHIOPIC('\u1369', '\u1200', '\u1380') {

            @Override
            char getNumericBase();
        }
        ,
        KHMER('\u17e0', '\u1780', '\u1800'),
        MONGOLIAN('\u1810', '\u1800', '\u1900'),
        NKO('\u07c0', '\u07c0', '\u0800'),
        MYANMAR_SHAN('\u1090', '\u1000', '\u10a0'),
        LIMBU('\u1946', '\u1900', '\u1950'),
        NEW_TAI_LUE('\u19d0', '\u1980', '\u19e0'),
        BALINESE('\u1b50', '\u1b00', '\u1b80'),
        SUNDANESE('\u1bb0', '\u1b80', '\u1bc0'),
        LEPCHA('\u1c40', '\u1c00', '\u1c50'),
        OL_CHIKI('\u1c50', '\u1c50', '\u1c80'),
        VAI('\ua620', '\ua500', '\ua640'),
        SAURASHTRA('\ua8d0', '\ua880', '\ua8e0'),
        KAYAH_LI('\ua900', '\ua900', '\ua930'),
        CHAM('\uaa50', '\uaa00', '\uaa60'),
        TAI_THAM_HORA('\u1a80', '\u1a20', '\u1ab0'),
        TAI_THAM_THAM('\u1a90', '\u1a20', '\u1ab0'),
        JAVANESE('\ua9d0', '\ua980', '\ua9e0'),
        MEETEI_MAYEK('\uabf0', '\uabc0', '\uac00'),
        SINHALA('\u0de6', '\u0d80', '\u0e00'),
        MYANMAR_TAI_LAING('\ua9f0', '\ua9e0', '\uaa00');

        char getNumericBase();
    }

    public static final int EUROPEAN;

    public static final int ARABIC;

    public static final int EASTERN_ARABIC;

    public static final int DEVANAGARI;

    public static final int BENGALI;

    public static final int GURMUKHI;

    public static final int GUJARATI;

    public static final int ORIYA;

    public static final int TAMIL;

    public static final int TELUGU;

    public static final int KANNADA;

    public static final int MALAYALAM;

    public static final int THAI;

    public static final int LAO;

    public static final int TIBETAN;

    public static final int MYANMAR;

    public static final int ETHIOPIC;

    public static final int KHMER;

    public static final int MONGOLIAN;

    public static final int ALL_RANGES;

    public static NumericShaper getShaper(int singleRange);

    public static NumericShaper getShaper(Range singleRange);

    public static NumericShaper getContextualShaper(int ranges);

    public static NumericShaper getContextualShaper(Set<Range> ranges);

    public static NumericShaper getContextualShaper(int ranges, int defaultContext);

    public static NumericShaper getContextualShaper(Set<Range> ranges, Range defaultContext);

    public void shape(char[] text, int start, int count);

    public void shape(char[] text, int start, int count, int context);

    public void shape(char[] text, int start, int count, Range context);

    public boolean isContextual();

    public int getRanges();

    public Set<Range> getRangeSet();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public String toString();
}
