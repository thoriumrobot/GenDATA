/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1998, 2017, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package com.sun.tools.jdi;

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
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import com.sun.jdi.AbsentInformationException;
    @Positive
import com.sun.jdi.ClassNotLoadedException;
    @Positive
import com.sun.jdi.IncompatibleThreadStateException;
    @Positive
import com.sun.jdi.InternalException;
    @Positive
import com.sun.jdi.InvalidStackFrameException;
    @Positive
import com.sun.jdi.InvalidTypeException;
    @Positive
import com.sun.jdi.LocalVariable;
    @Positive
import com.sun.jdi.Location;
    @Positive
import com.sun.jdi.ObjectReference;
    @Positive
import com.sun.jdi.StackFrame;
    @Positive
import com.sun.jdi.ThreadReference;
    @Positive
import com.sun.jdi.Value;
    @Positive
import com.sun.jdi.VirtualMachine;

    @Positive
public class StackFrameImpl extends MirrorImpl implements StackFrame, ThreadListener {

    @Positive
    public boolean threadResumable(ThreadAction action);

    @Positive
    void validateStackFrame();

    @Positive
    public Location location();

    @Positive
    public ThreadReference thread();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public ObjectReference thisObject();

    @Positive
    public List<LocalVariable> visibleVariables() throws AbsentInformationException;

    @Positive
    public LocalVariable visibleVariableByName(String name) throws AbsentInformationException;

    @Positive
    public Value getValue(LocalVariable variable);

    @Positive
    public Map<LocalVariable, Value> getValues(List<? extends LocalVariable> variables);

    @Positive
    public void setValue(LocalVariable variableIntf, Value valueIntf) throws InvalidTypeException, ClassNotLoadedException;

    @Positive
    public List<Value> getArgumentValues();

    @Positive
    void pop() throws IncompatibleThreadStateException;

    @Positive
    public String toString();
    @Positive
}
