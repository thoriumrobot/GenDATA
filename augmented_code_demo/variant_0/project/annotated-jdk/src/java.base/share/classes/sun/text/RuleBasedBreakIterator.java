/*
    @Positive
 * Copyright (c) 1999, 2016, Oracle and/or its affiliates. All rights reserved.
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
package sun.text;

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
import java.nio.BufferUnderflowException;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.text.BreakIterator;
    @Positive
import java.text.CharacterIterator;
    @Positive
import java.text.StringCharacterIterator;
    @Positive
import java.util.MissingResourceException;
    @Positive
import sun.text.CompactByteArray;
    @Positive
import sun.text.SupplementaryCharacterData;

    @Positive
public class RuleBasedBreakIterator extends BreakIterator {

    @Positive
    protected static final byte IGNORE;

    @Positive
    public RuleBasedBreakIterator(String ruleFile, byte[] ruleData) {
    @Positive
    }

    @Positive
    void validateRuleData(String ruleFile, ByteBuffer bb);

    @Positive
    byte[] getAdditionalData();

    @Positive
    void setAdditionalData(byte[] b);

    @Positive
    @Override
    @Positive
    public Object clone();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object that);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public int first();

    @Positive
    @Override
    @Positive
    public int last();

    @Positive
    @Override
    @Positive
    public int next(int n);

    @Positive
    @Override
    @Positive
    public int next();

    @Positive
    @Override
    @Positive
    public int previous();

    @Positive
    int getCurrent();

    @Positive
    int getNext();

    @Positive
    protected static final void checkOffset(int offset, CharacterIterator text);

    @Positive
    @Override
    @Positive
    public int following(int offset);

    @Positive
    @Override
    @Positive
    public int preceding(int offset);

    @Positive
    @Override
    @Positive
    public boolean isBoundary(int offset);

    @Positive
    @Override
    @Positive
    public int current();

    @Positive
    @Override
    @Positive
    public CharacterIterator getText();

    @Positive
    @Override
    @Positive
    public void setText(CharacterIterator newText);

    @Positive
    protected int handleNext();

    @Positive
    protected int handlePrevious();

    @Positive
    protected int lookupCategory(int c);

    @Positive
    protected int lookupState(int state, int category);

    @Positive
    protected int lookupBackwardState(int state, int category);

    @Positive
    private static final class SafeCharIterator implements CharacterIterator, Cloneable {

    @Positive
        @Override
    @Positive
        public char first();

    @Positive
        @Override
    @Positive
        public char last();

    @Positive
        @Override
    @Positive
        public char current();

    @Positive
        @Override
    @Positive
        public char next();

    @Positive
        @Override
    @Positive
        public char previous();

    @Positive
        @Override
    @Positive
        public char setIndex(int i);

    @Positive
        @Override
    @Positive
        public int getBeginIndex();

    @Positive
        @Override
    @Positive
        public int getEndIndex();

    @Positive
        @Override
    @Positive
        public int getIndex();

    @Positive
        @Override
    @Positive
        public Object clone();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
