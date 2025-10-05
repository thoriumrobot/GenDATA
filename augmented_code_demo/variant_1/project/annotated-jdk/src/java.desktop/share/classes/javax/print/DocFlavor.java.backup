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
package javax.print;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;

@SuppressWarnings("removal")
public class DocFlavor implements Serializable, Cloneable {

    public static final String hostEncoding;

    public DocFlavor(String mimeType, String className) {
    }

    public String getMimeType();

    public String getMediaType();

    public String getMediaSubtype();

    public String getParameter(String paramName);

    public String getRepresentationClassName();

    public String toString();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public static class BYTE_ARRAY extends DocFlavor {

        public BYTE_ARRAY(String mimeType) {
        }

        public static final BYTE_ARRAY TEXT_PLAIN_HOST;

        public static final BYTE_ARRAY TEXT_PLAIN_UTF_8;

        public static final BYTE_ARRAY TEXT_PLAIN_UTF_16;

        public static final BYTE_ARRAY TEXT_PLAIN_UTF_16BE;

        public static final BYTE_ARRAY TEXT_PLAIN_UTF_16LE;

        public static final BYTE_ARRAY TEXT_PLAIN_US_ASCII;

        public static final BYTE_ARRAY TEXT_HTML_HOST;

        public static final BYTE_ARRAY TEXT_HTML_UTF_8;

        public static final BYTE_ARRAY TEXT_HTML_UTF_16;

        public static final BYTE_ARRAY TEXT_HTML_UTF_16BE;

        public static final BYTE_ARRAY TEXT_HTML_UTF_16LE;

        public static final BYTE_ARRAY TEXT_HTML_US_ASCII;

        public static final BYTE_ARRAY PDF;

        public static final BYTE_ARRAY POSTSCRIPT;

        public static final BYTE_ARRAY PCL;

        public static final BYTE_ARRAY GIF;

        public static final BYTE_ARRAY JPEG;

        public static final BYTE_ARRAY PNG;

        public static final BYTE_ARRAY AUTOSENSE;
    }

    public static class INPUT_STREAM extends DocFlavor {

        public INPUT_STREAM(String mimeType) {
        }

        public static final INPUT_STREAM TEXT_PLAIN_HOST;

        public static final INPUT_STREAM TEXT_PLAIN_UTF_8;

        public static final INPUT_STREAM TEXT_PLAIN_UTF_16;

        public static final INPUT_STREAM TEXT_PLAIN_UTF_16BE;

        public static final INPUT_STREAM TEXT_PLAIN_UTF_16LE;

        public static final INPUT_STREAM TEXT_PLAIN_US_ASCII;

        public static final INPUT_STREAM TEXT_HTML_HOST;

        public static final INPUT_STREAM TEXT_HTML_UTF_8;

        public static final INPUT_STREAM TEXT_HTML_UTF_16;

        public static final INPUT_STREAM TEXT_HTML_UTF_16BE;

        public static final INPUT_STREAM TEXT_HTML_UTF_16LE;

        public static final INPUT_STREAM TEXT_HTML_US_ASCII;

        public static final INPUT_STREAM PDF;

        public static final INPUT_STREAM POSTSCRIPT;

        public static final INPUT_STREAM PCL;

        public static final INPUT_STREAM GIF;

        public static final INPUT_STREAM JPEG;

        public static final INPUT_STREAM PNG;

        public static final INPUT_STREAM AUTOSENSE;
    }

    public static class URL extends DocFlavor {

        public URL(String mimeType) {
        }

        public static final URL TEXT_PLAIN_HOST;

        public static final URL TEXT_PLAIN_UTF_8;

        public static final URL TEXT_PLAIN_UTF_16;

        public static final URL TEXT_PLAIN_UTF_16BE;

        public static final URL TEXT_PLAIN_UTF_16LE;

        public static final URL TEXT_PLAIN_US_ASCII;

        public static final URL TEXT_HTML_HOST;

        public static final URL TEXT_HTML_UTF_8;

        public static final URL TEXT_HTML_UTF_16;

        public static final URL TEXT_HTML_UTF_16BE;

        public static final URL TEXT_HTML_UTF_16LE;

        public static final URL TEXT_HTML_US_ASCII;

        public static final URL PDF;

        public static final URL POSTSCRIPT;

        public static final URL PCL;

        public static final URL GIF;

        public static final URL JPEG;

        public static final URL PNG;

        public static final URL AUTOSENSE;
    }

    public static class CHAR_ARRAY extends DocFlavor {

        public CHAR_ARRAY(String mimeType) {
        }

        public static final CHAR_ARRAY TEXT_PLAIN;

        public static final CHAR_ARRAY TEXT_HTML;
    }

    public static class STRING extends DocFlavor {

        public STRING(String mimeType) {
        }

        public static final STRING TEXT_PLAIN;

        public static final STRING TEXT_HTML;
    }

    public static class READER extends DocFlavor {

        public READER(String mimeType) {
        }

        public static final READER TEXT_PLAIN;

        public static final READER TEXT_HTML;
    }

    public static class SERVICE_FORMATTED extends DocFlavor {

        public SERVICE_FORMATTED(String className) {
        }

        public static final SERVICE_FORMATTED RENDERABLE_IMAGE;

        public static final SERVICE_FORMATTED PRINTABLE;

        public static final SERVICE_FORMATTED PAGEABLE;
    }
}
