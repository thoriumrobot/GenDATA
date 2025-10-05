/*
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.provider;

import org.checkerframework.dataflow.qual.Pure;
import java.io.*;
import java.security.GeneralSecurityException;
import java.security.Principal;
import java.util.*;
import javax.security.auth.x500.X500Principal;
import sun.security.util.Debug;
import sun.security.util.PropertyExpander;
import sun.security.util.LocalizedMessage;

public class PolicyParser {

    public PolicyParser() {
    }

    public PolicyParser(boolean expandProp) {
    }

    public void read(Reader policy) throws ParsingException, IOException;

    public void add(GrantEntry ge);

    public void replace(GrantEntry origGe, GrantEntry newGe);

    public boolean remove(GrantEntry ge);

    public String getKeyStoreUrl();

    public void setKeyStoreUrl(String url);

    public String getKeyStoreType();

    public void setKeyStoreType(String type);

    public String getKeyStoreProvider();

    public void setKeyStoreProvider(String provider);

    public String getStorePassURL();

    public void setStorePassURL(String storePassURL);

    public Enumeration<GrantEntry> grantElements();

    public Collection<DomainEntry> getDomainEntries();

    public void write(Writer policy);

    public static class GrantEntry {

        public String signedBy;

        public String codeBase;

        public LinkedList<PrincipalEntry> principals;

        public Vector<PermissionEntry> permissionEntries;

        public GrantEntry() {
        }

        public GrantEntry(String signedBy, String codeBase) {
        }

        public void add(PermissionEntry pe);

        public boolean remove(PrincipalEntry pe);

        public boolean remove(PermissionEntry pe);

        @Pure
        public boolean contains(PrincipalEntry pe);

        @Pure
        public boolean contains(PermissionEntry pe);

        public Enumeration<PermissionEntry> permissionElements();

        public void write(PrintWriter out);

        public Object clone();
    }

    public static class PrincipalEntry implements Principal {

        public static final String WILDCARD_CLASS;

        public static final String WILDCARD_NAME;

        public static final String REPLACE_NAME;

        public PrincipalEntry(String principalClass, String principalName) {
        }

        boolean isWildcardName();

        boolean isWildcardClass();

        boolean isReplaceName();

        public String getPrincipalClass();

        public String getPrincipalName();

        public String getDisplayClass();

        public String getDisplayName();

        public String getDisplayName(boolean addQuote);

        @Override
        public String getName();

        @Override
        public String toString();

        @Override
        public boolean equals(Object obj);

        @Override
        public int hashCode();

        public void write(PrintWriter out);
    }

    public static class PermissionEntry {

        public String permission;

        public String name;

        public String action;

        public String signedBy;

        public PermissionEntry() {
        }

        public PermissionEntry(String permission, String name, String action) {
        }

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object obj);

        public void write(PrintWriter out);
    }

    static class DomainEntry {

        String getName();

        Map<String, String> getProperties();

        Collection<KeyStoreEntry> getEntries();

        void add(KeyStoreEntry entry) throws ParsingException;

        @Override
        public String toString();
    }

    static class KeyStoreEntry {

        String getName();

        Map<String, String> getProperties();

        @Override
        public String toString();
    }

    public static class ParsingException extends GeneralSecurityException {

        public ParsingException(String msg) {
        }

        public ParsingException(String msg, LocalizedMessage localizedMsg, Object[] source) {
        }

        public ParsingException(int line, String msg) {
        }

        public ParsingException(int line, String expect, String actual) {
        }

        public String getNonlocalizedMessage();
    }

    public static void main(String[] arg) throws Exception;
}
