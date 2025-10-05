/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2013, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.jndi.ldap;

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
import java.util.Enumeration;
    @Positive
import java.util.Vector;
    @Positive
import java.util.Locale;
    @Positive
import javax.naming.*;
    @Positive
import javax.naming.directory.Attributes;
    @Positive
import javax.naming.directory.Attribute;
    @Positive
import javax.naming.directory.BasicAttributes;

    @Positive
public final class LdapName implements Name {

    @Positive
    public LdapName(String name) throws InvalidNameException {
    @Positive
    }

    @Positive
    public Object clone();

    @Positive
    public String toString();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int compareTo(Object obj);

    @Positive
    public int hashCode();

    @Positive
    public int size();

    @Positive
    public boolean isEmpty();

    @Positive
    public Enumeration<String> getAll();

    @Positive
    public String get(int pos);

    @Positive
    public Name getPrefix(int pos);

    @Positive
    public Name getSuffix(int pos);

    @Positive
    public boolean startsWith(Name n);

    @Positive
    public boolean endsWith(Name n);

    @Positive
    public void setValuesCaseSensitive(boolean caseSensitive);

    @Positive
    public Name addAll(Name suffix) throws InvalidNameException;

    @Positive
    public Name addAll(int pos, Name suffix) throws InvalidNameException;

    @Positive
    public Name add(String comp) throws InvalidNameException;

    @Positive
    public Name add(int pos, String comp) throws InvalidNameException;

    @Positive
    public Object remove(int pos) throws InvalidNameException;

    @Positive
    public static String escapeAttributeValue(Object val);

    @Positive
    public static Object unescapeAttributeValue(String val);

    @Positive
    static class DnParser {

    @Positive
        Vector<Rdn> getDn() throws InvalidNameException;

    @Positive
        Rdn getRdn() throws InvalidNameException;
    @Positive
    }

    @Positive
    static class Rdn {

    @Positive
        void add(TypeAndValue tv);

    @Positive
        public String toString();

    @Positive
        public boolean equals(Object obj);

    @Positive
        public int compareTo(Object obj);

    @Positive
        public int hashCode();

    @Positive
        Attributes toAttributes();
    @Positive
    }

    @Positive
    static class TypeAndValue {

    @Positive
        public String toString();

    @Positive
        public int compareTo(Object obj);

    @Positive
        public boolean equals(Object obj);

    @Positive
        public int hashCode();

    @Positive
        String getType();

    @Positive
        Object getUnescapedValue();

    @Positive
        static String escapeValue(Object val);

    @Positive
        static Object unescapeValue(String val);
    @Positive
    }
    @Positive
}
