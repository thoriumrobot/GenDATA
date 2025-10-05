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
package javax.print;

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
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;

    @Positive
@SuppressWarnings("removal")
    @Positive
public class DocFlavor implements Serializable, Cloneable {

    @Positive
    public static final String hostEncoding;

    @Positive
    public DocFlavor(String mimeType, String className) {
    @Positive
    }

    @Positive
    public String getMimeType();

    @Positive
    public String getMediaType();

    @Positive
    public String getMediaSubtype();

    @Positive
    public String getParameter(String paramName);

    @Positive
    public String getRepresentationClassName();

    @Positive
    public String toString();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public static class BYTE_ARRAY extends DocFlavor {

    @Positive
        public BYTE_ARRAY(String mimeType) {
    @Positive
        }

    @Positive
        public static final BYTE_ARRAY TEXT_PLAIN_HOST;

    @Positive
        public static final BYTE_ARRAY TEXT_PLAIN_UTF_8;

    @Positive
        public static final BYTE_ARRAY TEXT_PLAIN_UTF_16;

    @Positive
        public static final BYTE_ARRAY TEXT_PLAIN_UTF_16BE;

    @Positive
        public static final BYTE_ARRAY TEXT_PLAIN_UTF_16LE;

    @Positive
        public static final BYTE_ARRAY TEXT_PLAIN_US_ASCII;

    @Positive
        public static final BYTE_ARRAY TEXT_HTML_HOST;

    @Positive
        public static final BYTE_ARRAY TEXT_HTML_UTF_8;

    @Positive
        public static final BYTE_ARRAY TEXT_HTML_UTF_16;

    @Positive
        public static final BYTE_ARRAY TEXT_HTML_UTF_16BE;

    @Positive
        public static final BYTE_ARRAY TEXT_HTML_UTF_16LE;

    @Positive
        public static final BYTE_ARRAY TEXT_HTML_US_ASCII;

    @Positive
        public static final BYTE_ARRAY PDF;

    @Positive
        public static final BYTE_ARRAY POSTSCRIPT;

    @Positive
        public static final BYTE_ARRAY PCL;

    @Positive
        public static final BYTE_ARRAY GIF;

    @Positive
        public static final BYTE_ARRAY JPEG;

    @Positive
        public static final BYTE_ARRAY PNG;

    @Positive
        public static final BYTE_ARRAY AUTOSENSE;
    @Positive
    }

    @Positive
    public static class INPUT_STREAM extends DocFlavor {

    @Positive
        public INPUT_STREAM(String mimeType) {
    @Positive
        }

    @Positive
        public static final INPUT_STREAM TEXT_PLAIN_HOST;

    @Positive
        public static final INPUT_STREAM TEXT_PLAIN_UTF_8;

    @Positive
        public static final INPUT_STREAM TEXT_PLAIN_UTF_16;

    @Positive
        public static final INPUT_STREAM TEXT_PLAIN_UTF_16BE;

    @Positive
        public static final INPUT_STREAM TEXT_PLAIN_UTF_16LE;

    @Positive
        public static final INPUT_STREAM TEXT_PLAIN_US_ASCII;

    @Positive
        public static final INPUT_STREAM TEXT_HTML_HOST;

    @Positive
        public static final INPUT_STREAM TEXT_HTML_UTF_8;

    @Positive
        public static final INPUT_STREAM TEXT_HTML_UTF_16;

    @Positive
        public static final INPUT_STREAM TEXT_HTML_UTF_16BE;

    @Positive
        public static final INPUT_STREAM TEXT_HTML_UTF_16LE;

    @Positive
        public static final INPUT_STREAM TEXT_HTML_US_ASCII;

    @Positive
        public static final INPUT_STREAM PDF;

    @Positive
        public static final INPUT_STREAM POSTSCRIPT;

    @Positive
        public static final INPUT_STREAM PCL;

    @Positive
        public static final INPUT_STREAM GIF;

    @Positive
        public static final INPUT_STREAM JPEG;

    @Positive
        public static final INPUT_STREAM PNG;

    @Positive
        public static final INPUT_STREAM AUTOSENSE;
    @Positive
    }

    @Positive
    public static class URL extends DocFlavor {

    @Positive
        public URL(String mimeType) {
    @Positive
        }

    @Positive
        public static final URL TEXT_PLAIN_HOST;

    @Positive
        public static final URL TEXT_PLAIN_UTF_8;

    @Positive
        public static final URL TEXT_PLAIN_UTF_16;

    @Positive
        public static final URL TEXT_PLAIN_UTF_16BE;

    @Positive
        public static final URL TEXT_PLAIN_UTF_16LE;

    @Positive
        public static final URL TEXT_PLAIN_US_ASCII;

    @Positive
        public static final URL TEXT_HTML_HOST;

    @Positive
        public static final URL TEXT_HTML_UTF_8;

    @Positive
        public static final URL TEXT_HTML_UTF_16;

    @Positive
        public static final URL TEXT_HTML_UTF_16BE;

    @Positive
        public static final URL TEXT_HTML_UTF_16LE;

    @Positive
        public static final URL TEXT_HTML_US_ASCII;

    @Positive
        public static final URL PDF;

    @Positive
        public static final URL POSTSCRIPT;

    @Positive
        public static final URL PCL;

    @Positive
        public static final URL GIF;

    @Positive
        public static final URL JPEG;

    @Positive
        public static final URL PNG;

    @Positive
        public static final URL AUTOSENSE;
    @Positive
    }

    @Positive
    public static class CHAR_ARRAY extends DocFlavor {

    @Positive
        public CHAR_ARRAY(String mimeType) {
    @Positive
        }

    @Positive
        public static final CHAR_ARRAY TEXT_PLAIN;

    @Positive
        public static final CHAR_ARRAY TEXT_HTML;
    @Positive
    }

    @Positive
    public static class STRING extends DocFlavor {

    @Positive
        public STRING(String mimeType) {
    @Positive
        }

    @Positive
        public static final STRING TEXT_PLAIN;

    @Positive
        public static final STRING TEXT_HTML;
    @Positive
    }

    @Positive
    public static class READER extends DocFlavor {

    @Positive
        public READER(String mimeType) {
    @Positive
        }

    @Positive
        public static final READER TEXT_PLAIN;

    @Positive
        public static final READER TEXT_HTML;
    @Positive
    }

    @Positive
    public static class SERVICE_FORMATTED extends DocFlavor {

    @Positive
        public SERVICE_FORMATTED(String className) {
    @Positive
        }

    @Positive
        public static final SERVICE_FORMATTED RENDERABLE_IMAGE;

    @Positive
        public static final SERVICE_FORMATTED PRINTABLE;

    @Positive
        public static final SERVICE_FORMATTED PAGEABLE;
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
