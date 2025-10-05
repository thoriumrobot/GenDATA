/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.text;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.*;
import java.text.AttributedCharacterIterator.Attribute;

public class AttributedString {

    public AttributedString(String text) {
    }

    public AttributedString(String text, Map<? extends Attribute, ?> attributes) {
    }

    public AttributedString(AttributedCharacterIterator text) {
    }

    public AttributedString(AttributedCharacterIterator text, int beginIndex, int endIndex) {
    }

    public AttributedString(AttributedCharacterIterator text, int beginIndex, int endIndex, Attribute[] attributes) {
    }

    public void addAttribute(Attribute attribute, Object value);

    public void addAttribute(Attribute attribute, Object value, int beginIndex, int endIndex);

    public void addAttributes(Map<? extends Attribute, ?> attributes, int beginIndex, int endIndex);

    public AttributedCharacterIterator getIterator();

    public AttributedCharacterIterator getIterator(Attribute[] attributes);

    public AttributedCharacterIterator getIterator(Attribute[] attributes, int beginIndex, int endIndex);

    int length();

    private final class AttributedStringIterator implements AttributedCharacterIterator {

        public boolean equals(Object obj);

        public int hashCode();

        public Object clone();

        public char first();

        public char last();

        public char current();

        public char next();

        public char previous();

        public char setIndex(int position);

        public int getBeginIndex();

        public int getEndIndex();

        public int getIndex();

        public int getRunStart();

        public int getRunStart(Attribute attribute);

        public int getRunStart(Set<? extends Attribute> attributes);

        public int getRunLimit();

        public int getRunLimit(Attribute attribute);

        public int getRunLimit(Set<? extends Attribute> attributes);

        public Map<Attribute, Object> getAttributes();

        public Set<Attribute> getAllAttributeKeys();

        public Object getAttribute(Attribute attribute);
    }

    private final class AttributeMap extends AbstractMap<Attribute, Object> {

        public Set<Map.Entry<Attribute, Object>> entrySet();

        public Object get(Object key);
    }
}

class AttributeEntry implements Map.Entry<Attribute, Object> {

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public Attribute getKey();

    public Object getValue();

    public Object setValue(Object newValue);

    public int hashCode();

    public String toString();
}
