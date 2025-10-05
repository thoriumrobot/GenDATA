/*
 * Copyright (c) 1999, 2013, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.jndi.ldap;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.Enumeration;
import java.util.Vector;
import java.util.Locale;
import javax.naming.*;
import javax.naming.directory.Attributes;
import javax.naming.directory.Attribute;
import javax.naming.directory.BasicAttributes;

public final class LdapName implements Name {

    public LdapName(String name) throws InvalidNameException {
    }

    public Object clone();

    public String toString();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int compareTo(Object obj);

    public int hashCode();

    public int size();

    public boolean isEmpty();

    public Enumeration<String> getAll();

    public String get(int pos);

    public Name getPrefix(int pos);

    public Name getSuffix(int pos);

    public boolean startsWith(Name n);

    public boolean endsWith(Name n);

    public void setValuesCaseSensitive(boolean caseSensitive);

    public Name addAll(Name suffix) throws InvalidNameException;

    public Name addAll(int pos, Name suffix) throws InvalidNameException;

    public Name add(String comp) throws InvalidNameException;

    public Name add(int pos, String comp) throws InvalidNameException;

    public Object remove(int pos) throws InvalidNameException;

    public static String escapeAttributeValue(Object val);

    public static Object unescapeAttributeValue(String val);

    static class DnParser {

        Vector<Rdn> getDn() throws InvalidNameException;

        Rdn getRdn() throws InvalidNameException;
    }

    static class Rdn {

        void add(TypeAndValue tv);

        public String toString();

        public boolean equals(Object obj);

        public int compareTo(Object obj);

        public int hashCode();

        Attributes toAttributes();
    }

    static class TypeAndValue {

        public String toString();

        public int compareTo(Object obj);

        public boolean equals(Object obj);

        public int hashCode();

        String getType();

        Object getUnescapedValue();

        static String escapeValue(Object val);

        static Object unescapeValue(String val);
    }
}
