/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2015, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package jdk.internal.icu.impl;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.ArrayList;
    @Positive
import jdk.internal.icu.text.UTF16;
    @Positive
import jdk.internal.icu.text.UnicodeSet;
    @Positive
import jdk.internal.icu.text.UnicodeSet.SpanCondition;
    @Positive
import jdk.internal.icu.util.OutputInt;

    @Positive
public class UnicodeSetStringSpan {

    @Positive
    public static final int WITH_COUNT;

    @Positive
    public static final int FWD;

    @Positive
    public static final int BACK;

    @Positive
    public static final int CONTAINED;

    @Positive
    public static final int NOT_CONTAINED;

    @Positive
    public static final int ALL;

    @Positive
    public static final int FWD_UTF16_CONTAINED;

    @Positive
    public static final int FWD_UTF16_NOT_CONTAINED;

    @Positive
    public static final int BACK_UTF16_CONTAINED;

    @Positive
    public static final int BACK_UTF16_NOT_CONTAINED;

    @Positive
    public UnicodeSetStringSpan(final UnicodeSet set, final ArrayList<String> setStrings, int which) {
    @Positive
    }

    @Positive
    public boolean needsStringSpanUTF16();

    @Positive
    @Pure
    @Positive
    public boolean contains(int c);

    @Positive
    public int span(CharSequence s, int start, SpanCondition spanCondition);

    @Positive
    public int spanAndCount(CharSequence s, int start, SpanCondition spanCondition, OutputInt outCount);

    @Positive
    public synchronized int spanBack(CharSequence s, int length, SpanCondition spanCondition);

    @Positive
    static short makeSpanLengthByte(int spanLength);

    @Positive
    static boolean matches16CPB(CharSequence s, int start, int limit, final String t, int tlength);

    @Positive
    static int spanOne(final UnicodeSet set, CharSequence s, int start, int length);

    @Positive
    static int spanOneBack(final UnicodeSet set, CharSequence s, int length);

    @Positive
    private static final class OffsetList {

    @Positive
        public OffsetList() {
    @Positive
        }

    @Positive
        public void setMaxLength(int maxLength);

    @Positive
        public void clear();

    @Positive
        public boolean isEmpty();

    @Positive
        public void shift(int delta);

    @Positive
        public void addOffset(int offset);

    @Positive
        public void addOffsetAndCount(int offset, int count);

    @Positive
        @Pure
    @Positive
        public boolean containsOffset(int offset);

    @Positive
        public boolean hasCountAtOffset(int offset, int count);

    @Positive
        public int popMinimum(OutputInt outCount);
    @Positive
    }
    @Positive
}
