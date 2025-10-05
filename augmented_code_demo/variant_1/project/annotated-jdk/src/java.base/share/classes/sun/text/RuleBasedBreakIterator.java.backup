/*
 * Copyright (c) 1999, 2016, Oracle and/or its affiliates. All rights reserved.
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
package sun.text;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.text.BreakIterator;
import java.text.CharacterIterator;
import java.text.StringCharacterIterator;
import java.util.MissingResourceException;
import sun.text.CompactByteArray;
import sun.text.SupplementaryCharacterData;

public class RuleBasedBreakIterator extends BreakIterator {

    protected static final byte IGNORE;

    public RuleBasedBreakIterator(String ruleFile, byte[] ruleData) {
    }

    void validateRuleData(String ruleFile, ByteBuffer bb);

    byte[] getAdditionalData();

    void setAdditionalData(byte[] b);

    @Override
    public Object clone();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object that);

    @Override
    public String toString();

    @Override
    public int hashCode();

    @Override
    public int first();

    @Override
    public int last();

    @Override
    public int next(int n);

    @Override
    public int next();

    @Override
    public int previous();

    int getCurrent();

    int getNext();

    protected static final void checkOffset(int offset, CharacterIterator text);

    @Override
    public int following(int offset);

    @Override
    public int preceding(int offset);

    @Override
    public boolean isBoundary(int offset);

    @Override
    public int current();

    @Override
    public CharacterIterator getText();

    @Override
    public void setText(CharacterIterator newText);

    protected int handleNext();

    protected int handlePrevious();

    protected int lookupCategory(int c);

    protected int lookupState(int state, int category);

    protected int lookupBackwardState(int state, int category);

    private static final class SafeCharIterator implements CharacterIterator, Cloneable {

        @Override
        public char first();

        @Override
        public char last();

        @Override
        public char current();

        @Override
        public char next();

        @Override
        public char previous();

        @Override
        public char setIndex(int i);

        @Override
        public int getBeginIndex();

        @Override
        public int getEndIndex();

        @Override
        public int getIndex();

        @Override
        public Object clone();
    }
}
