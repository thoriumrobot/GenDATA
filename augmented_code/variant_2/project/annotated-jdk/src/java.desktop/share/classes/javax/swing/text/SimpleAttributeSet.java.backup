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
package javax.swing.text;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedHashMap;

@SuppressWarnings("serial")
public class SimpleAttributeSet implements MutableAttributeSet, Serializable, Cloneable {

    public static final AttributeSet EMPTY;

    public SimpleAttributeSet() {
    }

    public SimpleAttributeSet(AttributeSet source) {
    }

    public boolean isEmpty();

    public int getAttributeCount();

    public boolean isDefined(Object attrName);

    public boolean isEqual(AttributeSet attr);

    public AttributeSet copyAttributes();

    public Enumeration<?> getAttributeNames();

    public Object getAttribute(Object name);

    public boolean containsAttribute(Object name, Object value);

    public boolean containsAttributes(AttributeSet attributes);

    public void addAttribute(Object name, Object value);

    public void addAttributes(AttributeSet attributes);

    public void removeAttribute(Object name);

    public void removeAttributes(Enumeration<?> names);

    public void removeAttributes(AttributeSet attributes);

    public AttributeSet getResolveParent();

    public void setResolveParent(AttributeSet parent);

    @SuppressWarnings("unchecked")
    public Object clone();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public String toString();

    static class EmptyAttributeSet implements AttributeSet, Serializable {

        public int getAttributeCount();

        public boolean isDefined(Object attrName);

        public boolean isEqual(AttributeSet attr);

        public AttributeSet copyAttributes();

        public Object getAttribute(Object key);

        public Enumeration<?> getAttributeNames();

        public boolean containsAttribute(Object name, Object value);

        public boolean containsAttributes(AttributeSet attributes);

        public AttributeSet getResolveParent();

        public boolean equals(Object obj);

        public int hashCode();
    }
}
