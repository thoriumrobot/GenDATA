/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class OverloadedDynamicMethod {
/*
    @Positive
 * Copyright (c) 2010, 2021, Oracle and/or its affiliates. All rights reserved.
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
package jdk.dynalink.beans;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.lang.invoke.MethodHandle;
    @Positive
import java.lang.invoke.MethodType;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.text.Collator;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.IdentityHashMap;
    @Positive
import java.util.LinkedList;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import jdk.dynalink.CallSiteDescriptor;
    @Positive
import jdk.dynalink.SecureLookupSupplier;
    @Positive
import jdk.dynalink.beans.ApplicableOverloadedMethods.ApplicabilityTest;
    @Positive
import jdk.dynalink.internal.AccessControlContextFactory;
    @Positive
import jdk.dynalink.internal.InternalTypeUtilities;
    @Positive
import jdk.dynalink.linker.LinkerServices;

    @Positive
class OverloadedDynamicMethod extends DynamicMethod {

    @Positive
    @Override
    @Positive
    SingleDynamicMethod getMethodForExactParamTypes(final String paramTypes);

    @Positive
    @Override
    @Positive
    MethodHandle getInvocation(final CallSiteDescriptor callSiteDescriptor, final LinkerServices linkerServices);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean contains(final SingleDynamicMethod m);

    @Positive
    @Override
    @Positive
    public boolean isConstructor();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public void addMethod(final SingleDynamicMethod method);
    @Positive
}

}