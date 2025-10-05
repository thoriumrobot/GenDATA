/*
 * Copyright (c) 2005, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.internal.icu.text;

import org.checkerframework.dataflow.qual.Pure;
import java.text.ParsePosition;
import java.util.ArrayList;
import java.util.TreeSet;
import jdk.internal.icu.impl.BMPSet;
import jdk.internal.icu.impl.UCharacterProperty;
import jdk.internal.icu.impl.UnicodeSetStringSpan;
import jdk.internal.icu.impl.Utility;
import jdk.internal.icu.lang.UCharacter;
import jdk.internal.icu.util.OutputInt;
import jdk.internal.icu.util.VersionInfo;

public class UnicodeSet {

    public static final int MIN_VALUE;

    public static final int MAX_VALUE;

    public UnicodeSet(int start, int end) {
    }

    public UnicodeSet(String pattern) {
    }

    public UnicodeSet set(UnicodeSet other);

    public int size();

    public final UnicodeSet add(int c);

    public final UnicodeSet add(CharSequence s);

    public UnicodeSet complement(int start, int end);

    @Pure
    public boolean contains(int c);

    public UnicodeSet retainAll(UnicodeSet c);

    public UnicodeSet clear();

    public int getRangeCount();

    public int getRangeStart(int index);

    public int getRangeEnd(int index);

    private static interface Filter {

        @Pure
        boolean contains(int codePoint);
    }

    private static class VersionFilter implements Filter {

        @Pure
        public boolean contains(int ch);
    }

    public boolean isFrozen();

    public UnicodeSet freeze();

    public int span(CharSequence s, SpanCondition spanCondition);

    public int span(CharSequence s, int start, SpanCondition spanCondition);

    public int spanAndCount(CharSequence s, int start, SpanCondition spanCondition, OutputInt outCount);

    public int spanBack(CharSequence s, int fromIndex, SpanCondition spanCondition);

    public UnicodeSet cloneAsThawed();

    public enum SpanCondition {

        NOT_CONTAINED, CONTAINED, SIMPLE
    }
}
