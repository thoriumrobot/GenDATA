/*
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
package jdk.internal.org.objectweb.asm.commons;

import org.checkerframework.dataflow.qual.Pure;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import jdk.internal.org.objectweb.asm.ConstantDynamic;
import jdk.internal.org.objectweb.asm.Handle;
import jdk.internal.org.objectweb.asm.Label;
import jdk.internal.org.objectweb.asm.MethodVisitor;
import jdk.internal.org.objectweb.asm.Opcodes;
import jdk.internal.org.objectweb.asm.Type;

public abstract class AdviceAdapter extends GeneratorAdapter implements Opcodes {

    protected int methodAccess;

    protected String methodDesc;

    protected AdviceAdapter(final int api, final MethodVisitor methodVisitor, final int access, final String name, final String descriptor) {
    }

    @Override
    public void visitCode();

    @Override
    public void visitLabel(final Label label);

    @Override
    public void visitInsn(final int opcode);

    @Override
    public void visitVarInsn(final int opcode, final int var);

    @Override
    public void visitFieldInsn(final int opcode, final String owner, final String name, final String descriptor);

    @Override
    public void visitIntInsn(final int opcode, final int operand);

    @Override
    public void visitLdcInsn(final Object value);

    @Override
    public void visitMultiANewArrayInsn(final String descriptor, final int numDimensions);

    @Override
    public void visitTypeInsn(final int opcode, final String type);

    @Override
    public void visitMethodInsn(final int opcodeAndSource, final String owner, final String name, final String descriptor, final boolean isInterface);

    @Override
    public void visitInvokeDynamicInsn(final String name, final String descriptor, final Handle bootstrapMethodHandle, final Object... bootstrapMethodArguments);

    @Override
    public void visitJumpInsn(final int opcode, final Label label);

    @Override
    public void visitLookupSwitchInsn(final Label dflt, final int[] keys, final Label[] labels);

    @Override
    public void visitTableSwitchInsn(final int min, final int max, final Label dflt, final Label... labels);

    @Override
    public void visitTryCatchBlock(final Label start, final Label end, final Label handler, final String type);

    protected void onMethodEnter();

    protected void onMethodExit(final int opcode);
}
