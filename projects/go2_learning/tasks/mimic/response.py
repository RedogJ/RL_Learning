"""离线制动统计；不将 reset 或从未运动的样本算作急停成功。"""
import torch


def braking_metrics(state, commands, active, done, failed, dt, threshold=.1, hold_seconds=.3):
    state,commands,active,done,failed=[x.cpu() for x in (state,commands,active,done,failed)]
    hold_steps=max(1,round(hold_seconds/dt))
    events=[];late_idle=[];late_moving=[]
    for env in range(state.shape[1]):
        previous=None;event=None;slow=0;command_age=0.
        def finish(status):
            nonlocal event,slow
            if event is not None:
                event['status']=status;events.append(event)
            event=None;slow=0
        for step in range(state.shape[0]):
            if not bool(active[step,env]):
                finish('reset');previous=None;command_age=0.;continue
            command=float(commands[step,env,0]);speed=float(state[step,env,:2].norm())
            command_age=(command_age+dt if previous is not None and abs(command-previous)<.1 else 0.)
            if command_age>=1.:
                if abs(command)<.05:late_idle.append(speed)
                elif abs(command)>.5:late_moving.append(abs(float(state[step,env,0])-command))
            if previous is not None and abs(command-previous)>.1:
                finish('command_changed')
                if abs(command)<.05 and abs(previous)>.5:
                    event={'env':env,'step':step,'initial_speed':speed,'distance':0.,'elapsed':0.}
                    if speed<.5:finish('not_moving')
            if event is not None:
                event['distance']+=speed*dt
                event['elapsed']+=dt
                slow=slow+1 if speed<threshold else 0
                if bool(failed[step,env]):finish('failed')
                elif slow>=hold_steps:
                    event['settle_time']=max(0.,event['elapsed']-hold_seconds)
                    finish('stopped')
                elif bool(done[step,env]):finish('timeout')
            if bool(done[step,env]):previous=None
            else:previous=command
        finish('truncated')
    # 数据结束/正常超时截断的制动不作为失败；单独报告，避免窗口长短造成偏差。
    censored=('truncated','timeout','reset')
    eligible=[e for e in events if e['status'] not in ('not_moving',*censored)]
    successful=[e for e in eligible if e['status']=='stopped']
    moving=active & (commands[:,:,0].abs()>.5)
    idle=active & (commands[:,:,0].abs()<.05)
    return {'threshold_mps':threshold,'hold_seconds':hold_seconds,'events':events,
            'eligible_stops':len(eligible),'successful_stops':len(successful),
            'not_moving':sum(e['status']=='not_moving' for e in events),
            'censored_stops':sum(e['status'] in censored for e in events),
            'moving_command_mae':float((state[:,:,0]-commands[:,:,0]).abs()[moving].mean()) if moving.any() else None,
            'idle_speed_mean':float(state[:,:,:2].norm(dim=-1)[idle].mean()) if idle.any() else None,
            'idle_speed_after_1s':sum(late_idle)/len(late_idle) if late_idle else None,
            'moving_mae_after_1s':sum(late_moving)/len(late_moving) if late_moving else None,
            'success_fraction':len(successful)/len(eligible) if eligible else None,
            'mean_settle_time':sum(e['settle_time'] for e in successful)/len(successful) if successful else None,
            'mean_stop_distance':sum(e['distance'] for e in successful)/len(successful) if successful else None}
