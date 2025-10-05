/*
    @Positive
 * Copyright (c) 2015, 2019, Oracle and/or its affiliates. All rights reserved.
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
package sun.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import sun.nio.cs.ISO_8859_1;
    @Positive
import sun.nio.cs.UTF_8;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.CharBuffer;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.CharsetDecoder;
    @Positive
import java.nio.charset.CharsetEncoder;
    @Positive
import java.nio.charset.CoderResult;
    @Positive
import java.nio.charset.CodingErrorAction;
    @Positive
import java.nio.charset.StandardCharsets;
    @Positive
import java.util.Objects;

    @Positive
public class PropertyResourceBundleCharset extends Charset {

    @Positive
    public PropertyResourceBundleCharset(boolean strictUTF8) {
    @Positive
    }

    @Positive
    public PropertyResourceBundleCharset(String canonicalName, String[] aliases) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean contains(Charset cs);

    @Positive
    @Override
    @Positive
    public CharsetDecoder newDecoder();

    @Positive
    @Override
    @Positive
    public CharsetEncoder newEncoder();

    @Positive
    private final class PropertiesFileDecoder extends CharsetDecoder {

    @Positive
        protected PropertiesFileDecoder(Charset cs, float averageCharsPerByte, float maxCharsPerByte) {
    @Positive
        }

    @Positive
        protected CoderResult decodeLoop(ByteBuffer in, CharBuffer out);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
