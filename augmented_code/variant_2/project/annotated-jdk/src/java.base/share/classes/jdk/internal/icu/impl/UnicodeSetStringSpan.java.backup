/*
 * Copyright (c) 2015, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.internal.icu.impl;

import org.checkerframework.dataflow.qual.Pure;
import java.util.ArrayList;
import jdk.internal.icu.text.UTF16;
import jdk.internal.icu.text.UnicodeSet;
import jdk.internal.icu.text.UnicodeSet.SpanCondition;
import jdk.internal.icu.util.OutputInt;

public class UnicodeSetStringSpan {

    public static final int WITH_COUNT;

    public static final int FWD;

    public static final int BACK;

    public static final int CONTAINED;

    public static final int NOT_CONTAINED;

    public static final int ALL;

    public static final int FWD_UTF16_CONTAINED;

    public static final int FWD_UTF16_NOT_CONTAINED;

    public static final int BACK_UTF16_CONTAINED;

    public static final int BACK_UTF16_NOT_CONTAINED;

    public UnicodeSetStringSpan(final UnicodeSet set, final ArrayList<String> setStrings, int which) {
    }

    public boolean needsStringSpanUTF16();

    @Pure
    public boolean contains(int c);

    public int span(CharSequence s, int start, SpanCondition spanCondition);

    public int spanAndCount(CharSequence s, int start, SpanCondition spanCondition, OutputInt outCount);

    public synchronized int spanBack(CharSequence s, int length, SpanCondition spanCondition);

    static short makeSpanLengthByte(int spanLength);

    static boolean matches16CPB(CharSequence s, int start, int limit, final String t, int tlength);

    static int spanOne(final UnicodeSet set, CharSequence s, int start, int length);

    static int spanOneBack(final UnicodeSet set, CharSequence s, int length);

    private static final class OffsetList {

        public OffsetList() {
        }

        public void setMaxLength(int maxLength);

        public void clear();

        public boolean isEmpty();

        public void shift(int delta);

        public void addOffset(int offset);

        public void addOffsetAndCount(int offset, int count);

        @Pure
        public boolean containsOffset(int offset);

        public boolean hasCountAtOffset(int offset, int count);

        public int popMinimum(OutputInt outCount);
    }
}
