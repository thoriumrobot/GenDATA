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
package javax.print.attribute.standard;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.Serial;
import java.util.HashMap;
import java.util.Vector;
import javax.print.attribute.Attribute;
import javax.print.attribute.Size2DSyntax;

public class MediaSize extends Size2DSyntax implements Attribute {

    public MediaSize(float x, float y, int units) {
    }

    public MediaSize(int x, int y, int units) {
    }

    public MediaSize(float x, float y, int units, MediaSizeName media) {
    }

    public MediaSize(int x, int y, int units, MediaSizeName media) {
    }

    public MediaSizeName getMediaSizeName();

    public static MediaSize getMediaSizeForName(MediaSizeName media);

    public static MediaSizeName findMedia(float x, float y, int units);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object object);

    public final Class<? extends Attribute> getCategory();

    public final String getName();

    public static final class ISO {

        public static final MediaSize A0;

        public static final MediaSize A1;

        public static final MediaSize A2;

        public static final MediaSize A3;

        public static final MediaSize A4;

        public static final MediaSize A5;

        public static final MediaSize A6;

        public static final MediaSize A7;

        public static final MediaSize A8;

        public static final MediaSize A9;

        public static final MediaSize A10;

        public static final MediaSize B0;

        public static final MediaSize B1;

        public static final MediaSize B2;

        public static final MediaSize B3;

        public static final MediaSize B4;

        public static final MediaSize B5;

        public static final MediaSize B6;

        public static final MediaSize B7;

        public static final MediaSize B8;

        public static final MediaSize B9;

        public static final MediaSize B10;

        public static final MediaSize C3;

        public static final MediaSize C4;

        public static final MediaSize C5;

        public static final MediaSize C6;

        public static final MediaSize DESIGNATED_LONG;
    }

    public static final class JIS {

        public static final MediaSize B0;

        public static final MediaSize B1;

        public static final MediaSize B2;

        public static final MediaSize B3;

        public static final MediaSize B4;

        public static final MediaSize B5;

        public static final MediaSize B6;

        public static final MediaSize B7;

        public static final MediaSize B8;

        public static final MediaSize B9;

        public static final MediaSize B10;

        public static final MediaSize CHOU_1;

        public static final MediaSize CHOU_2;

        public static final MediaSize CHOU_3;

        public static final MediaSize CHOU_4;

        public static final MediaSize CHOU_30;

        public static final MediaSize CHOU_40;

        public static final MediaSize KAKU_0;

        public static final MediaSize KAKU_1;

        public static final MediaSize KAKU_2;

        public static final MediaSize KAKU_3;

        public static final MediaSize KAKU_4;

        public static final MediaSize KAKU_5;

        public static final MediaSize KAKU_6;

        public static final MediaSize KAKU_7;

        public static final MediaSize KAKU_8;

        public static final MediaSize KAKU_20;

        public static final MediaSize KAKU_A4;

        public static final MediaSize YOU_1;

        public static final MediaSize YOU_2;

        public static final MediaSize YOU_3;

        public static final MediaSize YOU_4;

        public static final MediaSize YOU_5;

        public static final MediaSize YOU_6;

        public static final MediaSize YOU_7;
    }

    public static final class NA {

        public static final MediaSize LETTER;

        public static final MediaSize LEGAL;

        public static final MediaSize NA_5X7;

        public static final MediaSize NA_8X10;

        public static final MediaSize NA_NUMBER_9_ENVELOPE;

        public static final MediaSize NA_NUMBER_10_ENVELOPE;

        public static final MediaSize NA_NUMBER_11_ENVELOPE;

        public static final MediaSize NA_NUMBER_12_ENVELOPE;

        public static final MediaSize NA_NUMBER_14_ENVELOPE;

        public static final MediaSize NA_6X9_ENVELOPE;

        public static final MediaSize NA_7X9_ENVELOPE;

        public static final MediaSize NA_9x11_ENVELOPE;

        public static final MediaSize NA_9x12_ENVELOPE;

        public static final MediaSize NA_10x13_ENVELOPE;

        public static final MediaSize NA_10x14_ENVELOPE;

        public static final MediaSize NA_10X15_ENVELOPE;
    }

    public static final class Engineering {

        public static final MediaSize A;

        public static final MediaSize B;

        public static final MediaSize C;

        public static final MediaSize D;

        public static final MediaSize E;
    }

    public static final class Other {

        public static final MediaSize EXECUTIVE;

        public static final MediaSize LEDGER;

        public static final MediaSize TABLOID;

        public static final MediaSize INVOICE;

        public static final MediaSize FOLIO;

        public static final MediaSize QUARTO;

        public static final MediaSize ITALY_ENVELOPE;

        public static final MediaSize MONARCH_ENVELOPE;

        public static final MediaSize PERSONAL_ENVELOPE;

        public static final MediaSize JAPANESE_POSTCARD;

        public static final MediaSize JAPANESE_DOUBLE_POSTCARD;
    }
}
