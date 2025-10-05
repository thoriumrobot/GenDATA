/*
    @Positive
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.provider;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.*;
    @Positive
import java.security.GeneralSecurityException;
    @Positive
import java.security.Principal;
    @Positive
import java.util.*;
    @Positive
import javax.security.auth.x500.X500Principal;
    @Positive
import sun.security.util.Debug;
    @Positive
import sun.security.util.PropertyExpander;
    @Positive
import sun.security.util.LocalizedMessage;

    @Positive
public class PolicyParser {

    @Positive
    public PolicyParser() {
    @Positive
    }

    @Positive
    public PolicyParser(boolean expandProp) {
    @Positive
    }

    @Positive
    public void read(Reader policy) throws ParsingException, IOException;

    @Positive
    public void add(GrantEntry ge);

    @Positive
    public void replace(GrantEntry origGe, GrantEntry newGe);

    @Positive
    public boolean remove(GrantEntry ge);

    @Positive
    public String getKeyStoreUrl();

    @Positive
    public void setKeyStoreUrl(String url);

    @Positive
    public String getKeyStoreType();

    @Positive
    public void setKeyStoreType(String type);

    @Positive
    public String getKeyStoreProvider();

    @Positive
    public void setKeyStoreProvider(String provider);

    @Positive
    public String getStorePassURL();

    @Positive
    public void setStorePassURL(String storePassURL);

    @Positive
    public Enumeration<GrantEntry> grantElements();

    @Positive
    public Collection<DomainEntry> getDomainEntries();

    @Positive
    public void write(Writer policy);

    @Positive
    public static class GrantEntry {

    @Positive
        public String signedBy;

    @Positive
        public String codeBase;

    @Positive
        public LinkedList<PrincipalEntry> principals;

    @Positive
        public Vector<PermissionEntry> permissionEntries;

    @Positive
        public GrantEntry() {
    @Positive
        }

    @Positive
        public GrantEntry(String signedBy, String codeBase) {
    @Positive
        }

    @Positive
        public void add(PermissionEntry pe);

    @Positive
        public boolean remove(PrincipalEntry pe);

    @Positive
        public boolean remove(PermissionEntry pe);

    @Positive
        @Pure
    @Positive
        public boolean contains(PrincipalEntry pe);

    @Positive
        @Pure
    @Positive
        public boolean contains(PermissionEntry pe);

    @Positive
        public Enumeration<PermissionEntry> permissionElements();

    @Positive
        public void write(PrintWriter out);

    @Positive
        public Object clone();
    @Positive
    }

    @Positive
    public static class PrincipalEntry implements Principal {

    @Positive
        public static final String WILDCARD_CLASS;

    @Positive
        public static final String WILDCARD_NAME;

    @Positive
        public static final String REPLACE_NAME;

    @Positive
        public PrincipalEntry(String principalClass, String principalName) {
    @Positive
        }

    @Positive
        boolean isWildcardName();

    @Positive
        boolean isWildcardClass();

    @Positive
        boolean isReplaceName();

    @Positive
        public String getPrincipalClass();

    @Positive
        public String getPrincipalName();

    @Positive
        public String getDisplayClass();

    @Positive
        public String getDisplayName();

    @Positive
        public String getDisplayName(boolean addQuote);

    @Positive
        @Override
    @Positive
        public String getName();

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        public void write(PrintWriter out);
    @Positive
    }

    @Positive
    public static class PermissionEntry {

    @Positive
        public String permission;

    @Positive
        public String name;

    @Positive
        public String action;

    @Positive
        public String signedBy;

    @Positive
        public PermissionEntry() {
    @Positive
        }

    @Positive
        public PermissionEntry(String permission, String name, String action) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);

    @Positive
        public void write(PrintWriter out);
    @Positive
    }

    @Positive
    static class DomainEntry {

    @Positive
        String getName();

    @Positive
        Map<String, String> getProperties();

    @Positive
        Collection<KeyStoreEntry> getEntries();

    @Positive
        void add(KeyStoreEntry entry) throws ParsingException;

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    static class KeyStoreEntry {

    @Positive
        String getName();

    @Positive
        Map<String, String> getProperties();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static class ParsingException extends GeneralSecurityException {

    @Positive
        public ParsingException(String msg) {
    @Positive
        }

    @Positive
        public ParsingException(String msg, LocalizedMessage localizedMsg, Object[] source) {
    @Positive
        }

    @Positive
        public ParsingException(int line, String msg) {
    @Positive
        }

    @Positive
        public ParsingException(int line, String expect, String actual) {
    @Positive
        }

    @Positive
        public String getNonlocalizedMessage();
    @Positive
    }

    @Positive
    public static void main(String[] arg) throws Exception;
    @Positive
}

// CFWR semantic augmentation - variant 1
