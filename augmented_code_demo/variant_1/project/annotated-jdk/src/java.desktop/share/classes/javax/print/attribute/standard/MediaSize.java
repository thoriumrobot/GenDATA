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
package javax.print.attribute.standard;

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
import java.io.Serial;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Vector;
    @Positive
import javax.print.attribute.Attribute;
    @Positive
import javax.print.attribute.Size2DSyntax;

    @Positive
public class MediaSize extends Size2DSyntax implements Attribute {

    @Positive
    public MediaSize(float x, float y, int units) {
    @Positive
    }

    @Positive
    public MediaSize(int x, int y, int units) {
    @Positive
    }

    @Positive
    public MediaSize(float x, float y, int units, MediaSizeName media) {
    @Positive
    }

    @Positive
    public MediaSize(int x, int y, int units, MediaSizeName media) {
    @Positive
    }

    @Positive
    public MediaSizeName getMediaSizeName();

    @Positive
    public static MediaSize getMediaSizeForName(MediaSizeName media);

    @Positive
    public static MediaSizeName findMedia(float x, float y, int units);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object object);

    @Positive
    public final Class<? extends Attribute> getCategory();

    @Positive
    public final String getName();

    @Positive
    public static final class ISO {

    @Positive
        public static final MediaSize A0;

    @Positive
        public static final MediaSize A1;

    @Positive
        public static final MediaSize A2;

    @Positive
        public static final MediaSize A3;

    @Positive
        public static final MediaSize A4;

    @Positive
        public static final MediaSize A5;

    @Positive
        public static final MediaSize A6;

    @Positive
        public static final MediaSize A7;

    @Positive
        public static final MediaSize A8;

    @Positive
        public static final MediaSize A9;

    @Positive
        public static final MediaSize A10;

    @Positive
        public static final MediaSize B0;

    @Positive
        public static final MediaSize B1;

    @Positive
        public static final MediaSize B2;

    @Positive
        public static final MediaSize B3;

    @Positive
        public static final MediaSize B4;

    @Positive
        public static final MediaSize B5;

    @Positive
        public static final MediaSize B6;

    @Positive
        public static final MediaSize B7;

    @Positive
        public static final MediaSize B8;

    @Positive
        public static final MediaSize B9;

    @Positive
        public static final MediaSize B10;

    @Positive
        public static final MediaSize C3;

    @Positive
        public static final MediaSize C4;

    @Positive
        public static final MediaSize C5;

    @Positive
        public static final MediaSize C6;

    @Positive
        public static final MediaSize DESIGNATED_LONG;
    @Positive
    }

    @Positive
    public static final class JIS {

    @Positive
        public static final MediaSize B0;

    @Positive
        public static final MediaSize B1;

    @Positive
        public static final MediaSize B2;

    @Positive
        public static final MediaSize B3;

    @Positive
        public static final MediaSize B4;

    @Positive
        public static final MediaSize B5;

    @Positive
        public static final MediaSize B6;

    @Positive
        public static final MediaSize B7;

    @Positive
        public static final MediaSize B8;

    @Positive
        public static final MediaSize B9;

    @Positive
        public static final MediaSize B10;

    @Positive
        public static final MediaSize CHOU_1;

    @Positive
        public static final MediaSize CHOU_2;

    @Positive
        public static final MediaSize CHOU_3;

    @Positive
        public static final MediaSize CHOU_4;

    @Positive
        public static final MediaSize CHOU_30;

    @Positive
        public static final MediaSize CHOU_40;

    @Positive
        public static final MediaSize KAKU_0;

    @Positive
        public static final MediaSize KAKU_1;

    @Positive
        public static final MediaSize KAKU_2;

    @Positive
        public static final MediaSize KAKU_3;

    @Positive
        public static final MediaSize KAKU_4;

    @Positive
        public static final MediaSize KAKU_5;

    @Positive
        public static final MediaSize KAKU_6;

    @Positive
        public static final MediaSize KAKU_7;

    @Positive
        public static final MediaSize KAKU_8;

    @Positive
        public static final MediaSize KAKU_20;

    @Positive
        public static final MediaSize KAKU_A4;

    @Positive
        public static final MediaSize YOU_1;

    @Positive
        public static final MediaSize YOU_2;

    @Positive
        public static final MediaSize YOU_3;

    @Positive
        public static final MediaSize YOU_4;

    @Positive
        public static final MediaSize YOU_5;

    @Positive
        public static final MediaSize YOU_6;

    @Positive
        public static final MediaSize YOU_7;
    @Positive
    }

    @Positive
    public static final class NA {

    @Positive
        public static final MediaSize LETTER;

    @Positive
        public static final MediaSize LEGAL;

    @Positive
        public static final MediaSize NA_5X7;

    @Positive
        public static final MediaSize NA_8X10;

    @Positive
        public static final MediaSize NA_NUMBER_9_ENVELOPE;

    @Positive
        public static final MediaSize NA_NUMBER_10_ENVELOPE;

    @Positive
        public static final MediaSize NA_NUMBER_11_ENVELOPE;

    @Positive
        public static final MediaSize NA_NUMBER_12_ENVELOPE;

    @Positive
        public static final MediaSize NA_NUMBER_14_ENVELOPE;

    @Positive
        public static final MediaSize NA_6X9_ENVELOPE;

    @Positive
        public static final MediaSize NA_7X9_ENVELOPE;

    @Positive
        public static final MediaSize NA_9x11_ENVELOPE;

    @Positive
        public static final MediaSize NA_9x12_ENVELOPE;

    @Positive
        public static final MediaSize NA_10x13_ENVELOPE;

    @Positive
        public static final MediaSize NA_10x14_ENVELOPE;

    @Positive
        public static final MediaSize NA_10X15_ENVELOPE;
    @Positive
    }

    @Positive
    public static final class Engineering {

    @Positive
        public static final MediaSize A;

    @Positive
        public static final MediaSize B;

    @Positive
        public static final MediaSize C;

    @Positive
        public static final MediaSize D;

    @Positive
        public static final MediaSize E;
    @Positive
    }

    @Positive
    public static final class Other {

    @Positive
        public static final MediaSize EXECUTIVE;

    @Positive
        public static final MediaSize LEDGER;

    @Positive
        public static final MediaSize TABLOID;

    @Positive
        public static final MediaSize INVOICE;

    @Positive
        public static final MediaSize FOLIO;

    @Positive
        public static final MediaSize QUARTO;

    @Positive
        public static final MediaSize ITALY_ENVELOPE;

    @Positive
        public static final MediaSize MONARCH_ENVELOPE;

    @Positive
        public static final MediaSize PERSONAL_ENVELOPE;

    @Positive
        public static final MediaSize JAPANESE_POSTCARD;

    @Positive
        public static final MediaSize JAPANESE_DOUBLE_POSTCARD;
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
