/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.module.ModuleDescriptor;
import java.lang.module.ModuleDescriptor.Exports;
import java.lang.module.ModuleDescriptor.Opens;
import java.lang.reflect.Member;
import java.io.FileDescriptor;
import java.io.File;
import java.io.FilePermission;
import java.net.InetAddress;
import java.net.SocketPermission;
import java.security.AccessControlContext;
import java.security.AccessController;
import java.security.Permission;
import java.security.PrivilegedAction;
import java.security.Security;
import java.security.SecurityPermission;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.PropertyPermission;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import jdk.internal.module.ModuleLoaderMap;
import jdk.internal.reflect.CallerSensitive;
import sun.security.util.SecurityConstants;

@AnnotatedFor({ "interning", "nullness" })
@Deprecated()
@UsesObjectEquals
public class SecurityManager {

    public SecurityManager() {
    }

    protected native Class<?>[] getClassContext();

    @SuppressWarnings("removal")
    public Object getSecurityContext();

    @SuppressWarnings("removal")
    public void checkPermission(Permission perm);

    @SuppressWarnings("removal")
    public void checkPermission(Permission perm, Object context);

    public void checkCreateClassLoader();

    public void checkAccess(Thread t);

    public void checkAccess(ThreadGroup g);

    public void checkExit(int status);

    public void checkExec(String cmd);

    public void checkLink(String lib);

    public void checkRead(FileDescriptor fd);

    public void checkRead(String file);

    public void checkRead(String file, Object context);

    public void checkWrite(FileDescriptor fd);

    public void checkWrite(String file);

    public void checkDelete(String file);

    public void checkConnect(String host, int port);

    public void checkConnect(String host, int port, Object context);

    public void checkListen(int port);

    public void checkAccept(String host, int port);

    public void checkMulticast(InetAddress maddr);

    @Deprecated()
    public void checkMulticast(InetAddress maddr, byte ttl);

    public void checkPropertiesAccess();

    public void checkPropertyAccess(String key);

    public void checkPrintJobAccess();

    static void addNonExportedPackages(ModuleLayer layer);

    static void invalidatePackageAccessCache();

    public void checkPackageAccess(String pkg);

    public void checkPackageDefinition(String pkg);

    public void checkSetFactory();

    public void checkSecurityAccess(String target);

    public ThreadGroup getThreadGroup();
}
