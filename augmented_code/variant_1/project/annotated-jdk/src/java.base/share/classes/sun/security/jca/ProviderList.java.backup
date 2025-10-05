/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.jca;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import java.util.*;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.security.Provider;
import java.security.Provider.Service;
import java.security.Security;

public final class ProviderList {

    @SuppressWarnings("removal")
    static ProviderList fromSecurityProperties();

    public static ProviderList add(ProviderList providerList, Provider p);

    public static ProviderList insertAt(ProviderList providerList, Provider p, int position);

    public static ProviderList remove(ProviderList providerList, String name);

    public static ProviderList newList(Provider... providers);

    ProviderList getJarList(String[] jarProvNames);

    public int size();

    Provider getProvider(int index);

    public List<Provider> providers();

    public Provider getProvider(String name);

    public int getIndex(String name);

    ProviderList removeInvalid();

    public Provider[] toArray();

    public String toString();

    public Service getService(String type, String name);

    public List<Service> getServices(String type, String algorithm);

    @Deprecated
    public List<Service> getServices(String type, List<String> algorithms);

    public List<Service> getServices(List<ServiceId> ids);

    private final class ServiceList extends AbstractList<Service> {

        public Service get(int index);

        public int size();

        public boolean isEmpty();

        public Iterator<Service> iterator();
    }

    static final class PreferredList {

        ArrayList<PreferredEntry> getAll(ServiceList s);

        ArrayList<PreferredEntry> getAll(String type, String algorithm);

        public PreferredEntry get(int i);

        public int size();

        public boolean add(PreferredEntry e);

        public String toString();
    }

    private static class PreferredEntry {

        boolean match(String t, String a);

        public String toString();
    }
}
