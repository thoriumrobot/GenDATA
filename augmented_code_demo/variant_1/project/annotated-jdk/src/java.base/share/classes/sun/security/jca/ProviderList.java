/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.jca;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.util.*;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.Provider;
    @Positive
import java.security.Provider.Service;
    @Positive
import java.security.Security;

    @Positive
public final class ProviderList {

    @Positive
    @SuppressWarnings("removal")
    @Positive
    static ProviderList fromSecurityProperties();

    @Positive
    public static ProviderList add(ProviderList providerList, Provider p);

    @Positive
    public static ProviderList insertAt(ProviderList providerList, Provider p, int position);

    @Positive
    public static ProviderList remove(ProviderList providerList, String name);

    @Positive
    public static ProviderList newList(Provider... providers);

    @Positive
    ProviderList getJarList(String[] jarProvNames);

    @Positive
    public int size();

    @Positive
    Provider getProvider(int index);

    @Positive
    public List<Provider> providers();

    @Positive
    public Provider getProvider(String name);

    @Positive
    public int getIndex(String name);

    @Positive
    ProviderList removeInvalid();

    @Positive
    public Provider[] toArray();

    @Positive
    public String toString();

    @Positive
    public Service getService(String type, String name);

    @Positive
    public List<Service> getServices(String type, String algorithm);

    @Positive
    @Deprecated
    @Positive
    public List<Service> getServices(String type, List<String> algorithms);

    @Positive
    public List<Service> getServices(List<ServiceId> ids);

    @Positive
    private final class ServiceList extends AbstractList<Service> {

    @Positive
        public Service get(int index);

    @Positive
        public int size();

    @Positive
        public boolean isEmpty();

    @Positive
        public Iterator<Service> iterator();
    @Positive
    }

    @Positive
    static final class PreferredList {

    @Positive
        ArrayList<PreferredEntry> getAll(ServiceList s);

    @Positive
        ArrayList<PreferredEntry> getAll(String type, String algorithm);

    @Positive
        public PreferredEntry get(int i);

    @Positive
        public int size();

    @Positive
        public boolean add(PreferredEntry e);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private static class PreferredEntry {

    @Positive
        boolean match(String t, String a);

    @Positive
        public String toString();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
