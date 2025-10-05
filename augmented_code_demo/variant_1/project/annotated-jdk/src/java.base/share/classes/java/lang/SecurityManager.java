/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.module.ModuleDescriptor;
    @Positive
import java.lang.module.ModuleDescriptor.Exports;
    @Positive
import java.lang.module.ModuleDescriptor.Opens;
    @Positive
import java.lang.reflect.Member;
    @Positive
import java.io.FileDescriptor;
    @Positive
import java.io.File;
    @Positive
import java.io.FilePermission;
    @Positive
import java.net.InetAddress;
    @Positive
import java.net.SocketPermission;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.Permission;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.Security;
    @Positive
import java.security.SecurityPermission;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.PropertyPermission;
    @Positive
import java.util.Set;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import jdk.internal.module.ModuleLoaderMap;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@Deprecated()
    @Positive
@UsesObjectEquals
    @Positive
public class SecurityManager {

    @Positive
    public SecurityManager() {
    @Positive
    }

    @Positive
    protected native Class<?>[] getClassContext();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public Object getSecurityContext();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public void checkPermission(Permission perm);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public void checkPermission(Permission perm, Object context);

    @Positive
    public void checkCreateClassLoader();

    @Positive
    public void checkAccess(Thread t);

    @Positive
    public void checkAccess(ThreadGroup g);

    @Positive
    public void checkExit(int status);

    @Positive
    public void checkExec(String cmd);

    @Positive
    public void checkLink(String lib);

    @Positive
    public void checkRead(FileDescriptor fd);

    @Positive
    public void checkRead(String file);

    @Positive
    public void checkRead(String file, Object context);

    @Positive
    public void checkWrite(FileDescriptor fd);

    @Positive
    public void checkWrite(String file);

    @Positive
    public void checkDelete(String file);

    @Positive
    public void checkConnect(String host, int port);

    @Positive
    public void checkConnect(String host, int port, Object context);

    @Positive
    public void checkListen(int port);

    @Positive
    public void checkAccept(String host, int port);

    @Positive
    public void checkMulticast(InetAddress maddr);

    @Positive
    @Deprecated()
    @Positive
    public void checkMulticast(InetAddress maddr, byte ttl);

    @Positive
    public void checkPropertiesAccess();

    @Positive
    public void checkPropertyAccess(String key);

    @Positive
    public void checkPrintJobAccess();

    @Positive
    static void addNonExportedPackages(ModuleLayer layer);

    @Positive
    static void invalidatePackageAccessCache();

    @Positive
    public void checkPackageAccess(String pkg);

    @Positive
    public void checkPackageDefinition(String pkg);

    @Positive
    public void checkSetFactory();

    @Positive
    public void checkSecurityAccess(String target);

    @Positive
    public ThreadGroup getThreadGroup();
    @Positive
}

// CFWR semantic augmentation - variant 1
