/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2018, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

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
import java.util.Locale;

    @Positive
public final class PageAttributes implements Cloneable {

    @Positive
    public static final class ColorType extends AttributeValue {

    @Positive
        public static final ColorType COLOR;

    @Positive
        public static final ColorType MONOCHROME;
    @Positive
    }

    @Positive
    public static final class MediaType extends AttributeValue {

    @Positive
        public static final MediaType ISO_4A0;

    @Positive
        public static final MediaType ISO_2A0;

    @Positive
        public static final MediaType ISO_A0;

    @Positive
        public static final MediaType ISO_A1;

    @Positive
        public static final MediaType ISO_A2;

    @Positive
        public static final MediaType ISO_A3;

    @Positive
        public static final MediaType ISO_A4;

    @Positive
        public static final MediaType ISO_A5;

    @Positive
        public static final MediaType ISO_A6;

    @Positive
        public static final MediaType ISO_A7;

    @Positive
        public static final MediaType ISO_A8;

    @Positive
        public static final MediaType ISO_A9;

    @Positive
        public static final MediaType ISO_A10;

    @Positive
        public static final MediaType ISO_B0;

    @Positive
        public static final MediaType ISO_B1;

    @Positive
        public static final MediaType ISO_B2;

    @Positive
        public static final MediaType ISO_B3;

    @Positive
        public static final MediaType ISO_B4;

    @Positive
        public static final MediaType ISO_B5;

    @Positive
        public static final MediaType ISO_B6;

    @Positive
        public static final MediaType ISO_B7;

    @Positive
        public static final MediaType ISO_B8;

    @Positive
        public static final MediaType ISO_B9;

    @Positive
        public static final MediaType ISO_B10;

    @Positive
        public static final MediaType JIS_B0;

    @Positive
        public static final MediaType JIS_B1;

    @Positive
        public static final MediaType JIS_B2;

    @Positive
        public static final MediaType JIS_B3;

    @Positive
        public static final MediaType JIS_B4;

    @Positive
        public static final MediaType JIS_B5;

    @Positive
        public static final MediaType JIS_B6;

    @Positive
        public static final MediaType JIS_B7;

    @Positive
        public static final MediaType JIS_B8;

    @Positive
        public static final MediaType JIS_B9;

    @Positive
        public static final MediaType JIS_B10;

    @Positive
        public static final MediaType ISO_C0;

    @Positive
        public static final MediaType ISO_C1;

    @Positive
        public static final MediaType ISO_C2;

    @Positive
        public static final MediaType ISO_C3;

    @Positive
        public static final MediaType ISO_C4;

    @Positive
        public static final MediaType ISO_C5;

    @Positive
        public static final MediaType ISO_C6;

    @Positive
        public static final MediaType ISO_C7;

    @Positive
        public static final MediaType ISO_C8;

    @Positive
        public static final MediaType ISO_C9;

    @Positive
        public static final MediaType ISO_C10;

    @Positive
        public static final MediaType ISO_DESIGNATED_LONG;

    @Positive
        public static final MediaType EXECUTIVE;

    @Positive
        public static final MediaType FOLIO;

    @Positive
        public static final MediaType INVOICE;

    @Positive
        public static final MediaType LEDGER;

    @Positive
        public static final MediaType NA_LETTER;

    @Positive
        public static final MediaType NA_LEGAL;

    @Positive
        public static final MediaType QUARTO;

    @Positive
        public static final MediaType A;

    @Positive
        public static final MediaType B;

    @Positive
        public static final MediaType C;

    @Positive
        public static final MediaType D;

    @Positive
        public static final MediaType E;

    @Positive
        public static final MediaType NA_10X15_ENVELOPE;

    @Positive
        public static final MediaType NA_10X14_ENVELOPE;

    @Positive
        public static final MediaType NA_10X13_ENVELOPE;

    @Positive
        public static final MediaType NA_9X12_ENVELOPE;

    @Positive
        public static final MediaType NA_9X11_ENVELOPE;

    @Positive
        public static final MediaType NA_7X9_ENVELOPE;

    @Positive
        public static final MediaType NA_6X9_ENVELOPE;

    @Positive
        public static final MediaType NA_NUMBER_9_ENVELOPE;

    @Positive
        public static final MediaType NA_NUMBER_10_ENVELOPE;

    @Positive
        public static final MediaType NA_NUMBER_11_ENVELOPE;

    @Positive
        public static final MediaType NA_NUMBER_12_ENVELOPE;

    @Positive
        public static final MediaType NA_NUMBER_14_ENVELOPE;

    @Positive
        public static final MediaType INVITE_ENVELOPE;

    @Positive
        public static final MediaType ITALY_ENVELOPE;

    @Positive
        public static final MediaType MONARCH_ENVELOPE;

    @Positive
        public static final MediaType PERSONAL_ENVELOPE;

    @Positive
        public static final MediaType A0;

    @Positive
        public static final MediaType A1;

    @Positive
        public static final MediaType A2;

    @Positive
        public static final MediaType A3;

    @Positive
        public static final MediaType A4;

    @Positive
        public static final MediaType A5;

    @Positive
        public static final MediaType A6;

    @Positive
        public static final MediaType A7;

    @Positive
        public static final MediaType A8;

    @Positive
        public static final MediaType A9;

    @Positive
        public static final MediaType A10;

    @Positive
        public static final MediaType B0;

    @Positive
        public static final MediaType B1;

    @Positive
        public static final MediaType B2;

    @Positive
        public static final MediaType B3;

    @Positive
        public static final MediaType B4;

    @Positive
        public static final MediaType ISO_B4_ENVELOPE;

    @Positive
        public static final MediaType B5;

    @Positive
        public static final MediaType ISO_B5_ENVELOPE;

    @Positive
        public static final MediaType B6;

    @Positive
        public static final MediaType B7;

    @Positive
        public static final MediaType B8;

    @Positive
        public static final MediaType B9;

    @Positive
        public static final MediaType B10;

    @Positive
        public static final MediaType C0;

    @Positive
        public static final MediaType ISO_C0_ENVELOPE;

    @Positive
        public static final MediaType C1;

    @Positive
        public static final MediaType ISO_C1_ENVELOPE;

    @Positive
        public static final MediaType C2;

    @Positive
        public static final MediaType ISO_C2_ENVELOPE;

    @Positive
        public static final MediaType C3;

    @Positive
        public static final MediaType ISO_C3_ENVELOPE;

    @Positive
        public static final MediaType C4;

    @Positive
        public static final MediaType ISO_C4_ENVELOPE;

    @Positive
        public static final MediaType C5;

    @Positive
        public static final MediaType ISO_C5_ENVELOPE;

    @Positive
        public static final MediaType C6;

    @Positive
        public static final MediaType ISO_C6_ENVELOPE;

    @Positive
        public static final MediaType C7;

    @Positive
        public static final MediaType ISO_C7_ENVELOPE;

    @Positive
        public static final MediaType C8;

    @Positive
        public static final MediaType ISO_C8_ENVELOPE;

    @Positive
        public static final MediaType C9;

    @Positive
        public static final MediaType ISO_C9_ENVELOPE;

    @Positive
        public static final MediaType C10;

    @Positive
        public static final MediaType ISO_C10_ENVELOPE;

    @Positive
        public static final MediaType ISO_DESIGNATED_LONG_ENVELOPE;

    @Positive
        public static final MediaType STATEMENT;

    @Positive
        public static final MediaType TABLOID;

    @Positive
        public static final MediaType LETTER;

    @Positive
        public static final MediaType NOTE;

    @Positive
        public static final MediaType LEGAL;

    @Positive
        public static final MediaType ENV_10X15;

    @Positive
        public static final MediaType ENV_10X14;

    @Positive
        public static final MediaType ENV_10X13;

    @Positive
        public static final MediaType ENV_9X12;

    @Positive
        public static final MediaType ENV_9X11;

    @Positive
        public static final MediaType ENV_7X9;

    @Positive
        public static final MediaType ENV_6X9;

    @Positive
        public static final MediaType ENV_9;

    @Positive
        public static final MediaType ENV_10;

    @Positive
        public static final MediaType ENV_11;

    @Positive
        public static final MediaType ENV_12;

    @Positive
        public static final MediaType ENV_14;

    @Positive
        public static final MediaType ENV_INVITE;

    @Positive
        public static final MediaType ENV_ITALY;

    @Positive
        public static final MediaType ENV_MONARCH;

    @Positive
        public static final MediaType ENV_PERSONAL;

    @Positive
        public static final MediaType INVITE;

    @Positive
        public static final MediaType ITALY;

    @Positive
        public static final MediaType MONARCH;

    @Positive
        public static final MediaType PERSONAL;
    @Positive
    }

    @Positive
    public static final class OrientationRequestedType extends AttributeValue {

    @Positive
        public static final OrientationRequestedType PORTRAIT;

    @Positive
        public static final OrientationRequestedType LANDSCAPE;
    @Positive
    }

    @Positive
    public static final class OriginType extends AttributeValue {

    @Positive
        public static final OriginType PHYSICAL;

    @Positive
        public static final OriginType PRINTABLE;
    @Positive
    }

    @Positive
    public static final class PrintQualityType extends AttributeValue {

    @Positive
        public static final PrintQualityType HIGH;

    @Positive
        public static final PrintQualityType NORMAL;

    @Positive
        public static final PrintQualityType DRAFT;
    @Positive
    }

    @Positive
    public PageAttributes() {
    @Positive
    }

    @Positive
    public PageAttributes(PageAttributes obj) {
    @Positive
    }

    @Positive
    public PageAttributes(ColorType color, MediaType media, OrientationRequestedType orientationRequested, OriginType origin, PrintQualityType printQuality, int[] printerResolution) {
    @Positive
    }

    @Positive
    public Object clone();

    @Positive
    public void set(PageAttributes obj);

    @Positive
    public ColorType getColor();

    @Positive
    public void setColor(ColorType color);

    @Positive
    public MediaType getMedia();

    @Positive
    public void setMedia(MediaType media);

    @Positive
    public void setMediaToDefault();

    @Positive
    public OrientationRequestedType getOrientationRequested();

    @Positive
    public void setOrientationRequested(OrientationRequestedType orientationRequested);

    @Positive
    public void setOrientationRequested(int orientationRequested);

    @Positive
    public void setOrientationRequestedToDefault();

    @Positive
    public OriginType getOrigin();

    @Positive
    public void setOrigin(OriginType origin);

    @Positive
    public PrintQualityType getPrintQuality();

    @Positive
    public void setPrintQuality(PrintQualityType printQuality);

    @Positive
    public void setPrintQuality(int printQuality);

    @Positive
    public void setPrintQualityToDefault();

    @Positive
    public int[] getPrinterResolution();

    @Positive
    public void setPrinterResolution(int[] printerResolution);

    @Positive
    public void setPrinterResolution(int printerResolution);

    @Positive
    public void setPrinterResolutionToDefault();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public String toString();
    @Positive
}
